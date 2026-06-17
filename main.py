import time

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI()


@app.websocket("/ws")
async def video_stream(websocket: WebSocket):
    await websocket.accept()
    print("\n[+] Android Client Connected!")

    window_name = "Vecta Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            # 1. Start the clock: Send current laptop time as a string ping
            laptop_start_time = int(time.time() * 1000)
            await websocket.send_text(str(laptop_start_time))

            # 2. Await the incoming frame from Android
            bytes_data = await websocket.receive_bytes()

            # 3. Extract the 8-byte echoed timestamp
            if len(bytes_data) > 8:
                echoed_timestamp = int.from_bytes(bytes_data[:8], byteorder="big")
                jpeg_bytes = bytes_data[8:]

                # 4. Accurate True Latency calculation (Round-trip halved)
                current_time = int(time.time() * 1000)
                true_latency = (current_time - echoed_timestamp) // 2

                nparr = np.frombuffer(jpeg_bytes, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                if frame is not None:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
                    # Display the true network transit latency
                    print(
                        f"\rTrue Network Latency: {true_latency:3d}ms | Frame Size: {len(jpeg_bytes):6d} bytes",
                        end="",
                    )
                    cv2.imshow(window_name, frame)

            cv2.waitKey(1)

    except WebSocketDisconnect:
        print("\n[-] Android Client Disconnected.")
    finally:
        cv2.destroyWindow(window_name)
