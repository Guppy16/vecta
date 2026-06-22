import sys
import json
import time
import sqlite3
import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()

DB_FILE = "telemetry.db"

# Initialize SQLite schema immediately on boot
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

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Vecta Brain is reachable!"}

@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] Android Client Connected! Booting Vision AI...")
    
    try:
        frames_received = 0
        while True:
            data = await websocket.receive_bytes()
            server_recv_time = int(time.time() * 1000) # Get current server epoch in ms
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
                
                # Calculate downstream one-way/approximate latency step
                latency_ms = server_recv_time - client_timestamp
                
                # Log telemetry cleanly to local SQLite file
                with sqlite3.connect(DB_FILE) as conn:
                    cursor = conn.cursor()
                    cursor.execute(
                        "INSERT INTO frame_telemetry (timestamp_ms, width, height, payload_kb, latency_ms) VALUES (?, ?, ?, ?, ?)",
                        (client_timestamp, w, h, payload_kb, latency_ms)
                    )
                    conn.commit()
                
                sys.stdout.write(f"\r[>] Frame #{frames_received:04d} | Size: {w}x{h} | Payload: {payload_kb:.1f} KB | Latency: {latency_ms}ms  ")
                sys.stdout.flush()
                
                # Send the response token back to phone UI
                response_data = {
                    "timestamp": client_timestamp,
                    "status": "processed"
                }
                await websocket.send_text(json.dumps(response_data))
            else:
                sys.stdout.write(f"\r[!] Frame #{frames_received:04d} | Failed to decode JPEG.         ")
                sys.stdout.flush()
                
    except WebSocketDisconnect:
        print("\n[-] Android Client Disconnected.")
    except Exception as e:
        print(f"\n[!] Unexpected Error: {e}")