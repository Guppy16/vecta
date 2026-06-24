import sys
import json
import time
import sqlite3
import os
from datetime import datetime
import numpy as np
import cv2
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
                latency_ms INTEGER
            )
        """)
        conn.commit()

init_db()

# --- NEW: YOLO-WORLD ZERO-SHOT SETUP ---
print("[*] Loading YOLO-World model...")
# Using the small (s) version for real-time speed over websockets
model = YOLO("assets/models/yolov8s-world.pt") 

# Define exactly what we are looking for. 
# YOLO-World will generate zero-shot embeddings for these words on the fly.
TARGET_CLASSES = ["keys", "keyboard", "fan"]
model.set_classes(TARGET_CLASSES)
print(f"[*] Vision Engine searching for: {TARGET_CLASSES}")


@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Vecta Brain is reachable!"}

@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] Android Client Connected! Booting Vision Pipeline...")
    
    try:
        frames_received = 0
        while True:
            data = await websocket.receive_bytes()
            server_recv_time = int(time.time() * 1000)
            frames_received += 1
            
            if len(data) < 8:
                continue
                
            client_timestamp = int.from_bytes(data[:8], byteorder='big')
            jpeg_bytes = data[8:]
            
            nparr = np.frombuffer(jpeg_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                h, w, c = img.shape
                payload_kb = len(jpeg_bytes) / 1024
                latency_ms = server_recv_time - client_timestamp
                
                # Run YOLO-World Inference
                results = model(img, verbose=False)
                
                target_found = False
                x_norm = 0.0
                y_norm = 0.0
                found_item = ""
                
                # Check for detections
                if len(results[0].boxes) > 0:
                    # Just grab the box with the highest confidence for the PoC
                    best_box = results[0].boxes[0]
                    class_id = int(best_box.cls[0])
                    found_item = TARGET_CLASSES[class_id]
                    
                    x_center, y_center = best_box.xywh[0][:2].tolist()
                    x_norm = x_center / w
                    y_norm = y_center / h
                    target_found = True
                
                # Log telemetry to our new timestamped directory
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO frame_telemetry (timestamp_ms, width, height, payload_kb, latency_ms) VALUES (?, ?, ?, ?, ?)",
                        (client_timestamp, w, h, payload_kb, latency_ms)
                    )
                    conn.commit()
                
                # Terminal UI
                detect_str = f"| FOUND: {found_item.upper()} " if target_found else "| Searching...        "
                sys.stdout.write(f"\r[>] Frame #{frames_received:04d} | Latency: {latency_ms}ms {detect_str}")
                sys.stdout.flush()
                
                # Send coordinates back to Android
                response_data = {
                    "timestamp": client_timestamp,
                    "status": "processed",
                    "target_found": target_found,
                    "item_name": found_item,
                    "x_norm": x_norm,
                    "y_norm": y_norm
                }
                await websocket.send_text(json.dumps(response_data))
            else:
                sys.stdout.write(f"\r[!] Frame #{frames_received:04d} | Failed to decode JPEG.         ")
                sys.stdout.flush()
                
    except WebSocketDisconnect:
        print("\n[-] Android Client Disconnected.")
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")