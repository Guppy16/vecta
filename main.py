import sys
import json
import time
import asyncio
import sqlite3
import os
from datetime import datetime
import numpy as np
import cv2
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from ultralytics import YOLO

app = FastAPI()

session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = os.path.join("logs", f"session_{session_timestamp}")
os.makedirs(LOG_DIR, exist_ok=True)
DB_FILE = os.path.join(LOG_DIR, "telemetry.db")
print(f"[*] Telemetry will be logged to: {DB_FILE}")


def init_db():
    with sqlite3.connect(DB_FILE) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS frame_telemetry (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ms INTEGER,
                width INTEGER,
                height INTEGER,
                payload_kb REAL,
                latency_ms INTEGER,      -- client-vs-server wall-clock difference (CLOCK SKEW, not true latency)
                infer_ms REAL,           -- our wall-clock around inference, GPU-synced (trustworthy)
                yolo_ms REAL,            -- Ultralytics' own internal inference time (cross-check)
                dropped_before INTEGER   -- frames discarded since last processed (backpressure signal)
            )
        """)
        conn.commit()


init_db()


def sync_device():
    """Block until queued GPU work is actually finished, so timing measures compute
    (not just async dispatch). No-op on CPU. torch.cuda also drives ROCm/HIP queues."""
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


# --- YOLO-WORLD ZERO-SHOT SETUP ---
print("[*] Loading YOLO-World model...")
model = YOLO("assets/models/yolov8x-world.pt")

TARGET_CLASSES = ["keys", "keyboard", "fan", "laptop", "water bottle", ]
model.set_classes(TARGET_CLASSES)

# Only place a marker above this confidence. Zero-shot throws low-confidence
# false positives constantly; without a floor you anchor on noise.
CONF_THRESHOLD = 0.25

print(f"[*] Vision Engine searching for: {TARGET_CLASSES}")
print(f"[*] Torch device: cuda_available={torch.cuda.is_available()} "
      f"device_name={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- WARMUP ---
# First inference compiles kernels (on ROCm: MIOpen) and can take many seconds.
# Doing it now means the user's first real scan is fast.
print("[*] Warming up model (first inference is always slow)...")
_t = time.time()
_ = model(np.zeros((480, 640, 3), dtype=np.uint8), verbose=False)
sync_device()
print(f"[*] Warmup complete in {(time.time() - _t) * 1000:.0f}ms. Ready.\n")


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Vecta Brain is reachable!"}


def run_inference(jpeg_bytes):
    """Decode + infer + GPU-sync, all in a worker thread so the event loop keeps
    draining frames. The sync_device() call is what makes infer_ms honest."""
    nparr = np.frombuffer(jpeg_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    results = model(img, verbose=False)
    sync_device()  # wait for the GPU to actually finish before we stop timing
    return img, results


@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("[+] Android Client Connected! Booting Vision Pipeline...")

    # --- LATEST-FRAME-WINS PLUMBING ---
    latest = {"data": None, "dropped": 0}
    new_frame = asyncio.Event()
    disconnected = asyncio.Event()

    async def receiver():
        try:
            while True:
                data = await websocket.receive_bytes()
                if latest["data"] is not None:
                    latest["dropped"] += 1
                latest["data"] = data
                new_frame.set()
        except WebSocketDisconnect:
            pass
        except Exception as e:
            print(f"\n[!] Receiver error: {e}")
        finally:
            disconnected.set()
            new_frame.set()

    recv_task = asyncio.create_task(receiver())

    db_conn = sqlite3.connect(DB_FILE)
    cursor = db_conn.cursor()
    frames_processed = 0

    try:
        while not disconnected.is_set():
            await new_frame.wait()
            new_frame.clear()

            data = latest["data"]
            latest["data"] = None
            dropped_before = latest["dropped"]
            latest["dropped"] = 0

            if data is None or len(data) < 8:
                continue

            frames_processed += 1
            client_timestamp = int.from_bytes(data[:8], byteorder='big')
            jpeg_bytes = data[8:]

            t0 = time.time()
            img, results = await asyncio.to_thread(run_inference, jpeg_bytes)
            infer_ms = (time.time() - t0) * 1000

            if img is None:
                sys.stdout.write(f"\r[!] Frame #{frames_processed:04d} | Failed to decode JPEG.         ")
                sys.stdout.flush()
                continue

            # Ultralytics' own GPU-synced timing, as an independent cross-check.
            speed = results[0].speed  # {'preprocess', 'inference', 'postprocess'} in ms
            yolo_ms = float(speed.get("inference", 0.0)) + float(speed.get("postprocess", 0.0))

            h, w = img.shape[:2]
            payload_kb = len(jpeg_bytes) / 1024
            latency_ms = int(time.time() * 1000) - client_timestamp  # skew, reference only

            target_found = False
            x_norm = 0.0
            y_norm = 0.0
            found_item = ""
            best_conf = 0.0

            boxes = results[0].boxes
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf
                best_idx = int(confs.argmax())
                best_conf = float(confs[best_idx])
                if best_conf >= CONF_THRESHOLD:
                    class_id = int(boxes.cls[best_idx])
                    found_item = TARGET_CLASSES[class_id]
                    x_center, y_center = boxes.xywh[best_idx][:2].tolist()
                    x_norm = x_center / w
                    y_norm = y_center / h
                    target_found = True

            cursor.execute(
                "INSERT INTO frame_telemetry "
                "(timestamp_ms, width, height, payload_kb, latency_ms, infer_ms, yolo_ms, dropped_before) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (client_timestamp, w, h, payload_kb, latency_ms, infer_ms, yolo_ms, dropped_before)
            )
            db_conn.commit()

            if target_found:
                detect_str = f"| FOUND: {found_item.upper()} ({best_conf:.2f}) "
            else:
                detect_str = "| Searching...           "
            sys.stdout.write(
                f"\r[>] #{frames_processed:04d} | infer={infer_ms:4.0f}ms (yolo={yolo_ms:4.0f}ms) "
                f"| dropped={dropped_before:2d} {detect_str}"
            )
            sys.stdout.flush()

            response_data = {
                "timestamp": client_timestamp,
                "status": "processed",
                "target_found": target_found,
                "item_name": found_item,
                "x_norm": x_norm,
                "y_norm": y_norm
            }
            await websocket.send_text(json.dumps(response_data))

    except WebSocketDisconnect:
        print("\n[-] Android Client Disconnected.")
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")
    finally:
        recv_task.cancel()
        db_conn.close()
        print("\n[-] Session closed.")