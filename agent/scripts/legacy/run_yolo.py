#!/usr/bin/env python3
"""
Run YOLO (Nano segmentation) on a local MP4 video and show live annotated frames
with rolling performance metrics printed in-place.
"""

import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# Directory to save assets and crops
ASSETS_DIR = Path("assets")
MODELS_DIR = ASSETS_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR = ASSETS_DIR / "captured_keyboards"
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)

# Path to the video file
VIDEO_PATH = "/home/guppy/git/vecta/assets/VID_20260617_201205.mp4"

# YOLO weights file
# MODEL_WEIGHTS = "yolo26m-seg.pt"
MODEL_WEIGHTS = MODELS_DIR / "yolov8l-world.pt"
# MODEL_WEIGHTS = "yolo26n-seg.onnx"


def main():
    # Basic checks
    if not Path(MODEL_WEIGHTS).exists():
        print(f"Warning: model weights not found at '{MODEL_WEIGHTS}'.")
        print(
            "If you don't have the weights locally the ultralytics package may try to download them,"
        )
        print("or set MODEL_WEIGHTS to a valid path.")

    print("Loading AI Model...")
    model = YOLO(MODEL_WEIGHTS)

    model.set_classes(["fan", "curtain", "world globe", "keyboard", "phone"])

    # --- START NATIVE PYTORCH OPTIMIZATIONS ---
    import torch

    # print("Pushing model to ROCm and converting to FP16...")
    # model.to("cuda")  # Move to the AMD GPU

    # print("Injecting Triton Graph Compiler...")
    # This fuses the PyTorch operations into a single optimized hardware kernel
    # model.model = torch.compile(model.model, backend="inductor", mode="reduce-overhead")
    # --- END NATIVE PYTORCH OPTIMIZATIONS ---

    print(f"Opening video: {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Failed to open video: {VIDEO_PATH}")
        return

    video_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Video Native Frame Rate: {video_fps} FPS")

    seen_keyboard_ids = set()

    window_name = "YOLO Live"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    prev_time = time.time()
    frame_count = 0
    fps = 0.0

    # Initialize a rolling window buffer for the last 100 latencies
    latency_buffer = deque(maxlen=100)

    print("Starting inference. Press 'q' or ESC to quit.")
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video or cannot read frame.")
                break

            # --- START LATENCY TIMING ---
            t_start = time.perf_counter()

            try:
                # results = model.track(
                #     frame,
                #     classes=[66],
                #     persist=True,
                #     verbose=False,
                #     # device="cpu",
                #     # half=True,
                # )
                # results = model.predict(
                #     frame,
                #     classes=[66],
                #     verbose=False,
                #     device=0,
                # )
                results = model.track(
                    frame,
                    # classes=[66],
                    persist=True,
                    verbose=False,
                    tracker="bytetrack.yaml",  # <--- FORCE BYTETRACK HERE
                    # conf=0.25,  # <--- IGNORE NOISE
                    device=0,
                    half=True,
                )
                # --- ADD THIS PROFILER PRINT ---
                # import torch

                # torch.cuda.synchronize()
                # speed = results[0].speed
                # print(
                #     f"\n[Profiler] CPU Pre: {speed['preprocess']:.1f}ms | GPU Infer: {speed['inference']:.1f}ms | CPU Post: {speed['postprocess']:.1f}ms"
                # )
            except Exception as e:
                print("\nInference error:", e)
                break

            t_end = time.perf_counter()

            # Store latency in milliseconds
            current_latency = (t_end - t_start) * 1000
            latency_buffer.append(current_latency)

            # Calculate metrics from the current buffer
            latencies = np.array(latency_buffer)
            mean_latency = np.mean(latencies)
            p95_latency = np.percentile(latencies, 95)

            # \r moves cursor to the start of the line; trailing spaces clear out old text artifacts
            print(
                f"\r[Buffer: {len(latency_buffer):>3}/100] "
                f"Rolling Mean: {mean_latency:>5.1f}ms | "
                f"P95 Latency: {p95_latency:>5.1f}ms    ",
                end="",
                flush=True,
            )
            # --- END LATENCY TIMING ---

            # Get annotated frame
            annotated_frame = results[0].plot()

            # Save crops for newly seen keyboard track IDs
            try:
                boxes_obj = results[0].boxes
                if hasattr(boxes_obj, "id") and boxes_obj.id is not None:
                    boxes = boxes_obj.xyxy.cpu().numpy().astype(int)
                    track_ids = boxes_obj.id.cpu().numpy().astype(int)
                    h, w = frame.shape[:2]
                    for box, track_id in zip(boxes, track_ids):
                        if int(track_id) not in seen_keyboard_ids:
                            x1, y1, x2, y2 = box
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(w, x2), min(h, y2)
                            crop = frame[y1:y2, x1:x2]
                            if crop.size > 0:
                                out_path = (
                                    CAPTURE_DIR / f"keyboard_id_{int(track_id)}.jpg"
                                )
                                cv2.imwrite(str(out_path), crop)
                                seen_keyboard_ids.add(int(track_id))

                                # Print alert on a new line so it doesn't break our rolling stats line
                                print(
                                    f"\n[AI] Captured new keyboard — saved {out_path}"
                                )
            except Exception as e:
                pass

            # Compute and display window FPS
            frame_count += 1
            now = time.time()
            elapsed = now - prev_time
            if elapsed >= 1.0:
                fps = frame_count / elapsed
                frame_count = 0
                prev_time = now

            try:
                cv2.putText(
                    annotated_frame,
                    f"FPS: {fps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )
            except Exception:
                pass

            cv2.imshow(window_name, annotated_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:
                print("\nExit requested by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
