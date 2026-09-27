"""Stand-in for the Android app: streams a video file over WebRTC and drives the
control channel, printing everything the server sends back.

    uv run python scripts/fake_phone.py --server http://127.0.0.1:8000 --video clip.mp4

Useful for testing the server end to end from a laptop, and as a reference for
what the phone must do (offer -> POST /rtc/offer -> answer, then messages).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import httpx
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer

from vecta.protocol import messages as m


async def run(server: str, video: str, seconds: float) -> None:
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    player = MediaPlayer(video, loop=True)
    pc.addTrack(player.video)
    channel = pc.createDataChannel("control")
    done = asyncio.Event()

    @channel.on("open")
    def on_open() -> None:
        send(m.SessionStart(device={"w": 1080, "h": 2400, "dpr": 2.5}))
        send(m.Ping(t=now_ms()))

    @channel.on("message")
    def on_message(data: str) -> None:
        msg = m.decode(data)
        match msg:
            case m.Pong(t=t):
                print(f"<- pong  rtt={now_ms() - t} ms")
            case m.PageRender(html=html, version=v):
                print(f"<- page.render v{v} ({len(html)} chars)\n{html}\n")
            case _:
                print(f"<- {msg}")

    def send(msg: m.Message) -> None:
        print(f"-> {msg.type}")
        channel.send(m.encode(msg))

    await pc.setLocalDescription(await pc.createOffer())
    async with httpx.AsyncClient(timeout=30) as http:
        r = await http.post(
            f"{server}/rtc/offer",
            json={"sdp": pc.localDescription.sdp, "type": pc.localDescription.type},
        )
        r.raise_for_status()
    answer = r.json()
    await pc.setRemoteDescription(RTCSessionDescription(sdp=answer["sdp"], type=answer["type"]))
    print(f"session {answer['session_id']}")

    # a scripted "user": set a task, wait for frames to flow, take a photo
    await asyncio.sleep(1.5)
    send(m.TaskSet(text="which almond butter is best value"))
    await asyncio.sleep(1.5)
    send(m.Capture(kind="photo", id="c1", ts=now_ms()))
    await asyncio.sleep(0.5)
    send(m.Ping(t=now_ms()))
    try:
        await asyncio.wait_for(done.wait(), timeout=seconds)
    except TimeoutError:
        pass
    await pc.close()


def now_ms() -> int:
    return int(time.time() * 1000)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--video", required=True, help="video file to stream (looped)")
    ap.add_argument("--seconds", type=float, default=5.0, help="how long to stay connected")
    ap.add_argument("-v", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.v else logging.WARNING)
    asyncio.run(run(args.server, args.video, args.seconds))


if __name__ == "__main__":
    main()
