"""HTTP entrypoint: WebRTC signalling, asset serving, and the message handler.

Until the Claude agent layer lands, the handler is a local VLM watch: set a
task and the box tells you when the camera sees it; ask a question and it
answers from recent frames.
"""

from __future__ import annotations

import io
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from vecta.protocol import messages as m
from vecta.server.frames import Frame
from vecta.server.rtc import Peer
from vecta.server.sessions import SessionStore
from vecta.server.vlm import Vlm
from vecta.server.watch import Watcher

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("VECTA_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VECTA_PORT", "8000"))
    data_dir: Path = Path(os.environ.get("VECTA_DATA_DIR", Path.home() / ".vecta"))


settings = Settings()
app = FastAPI(title="vecta")
store = SessionStore(settings.data_dir)
vlm = Vlm()
peers: dict[str, Peer] = {}
watchers: dict[str, Watcher] = {}


@app.get("/ping")
async def ping() -> dict[str, int]:
    return {"t": int(time.time() * 1000)}


@app.post("/rtc/offer")
async def rtc_offer(request: Request) -> dict[str, str]:
    """Answer the phone's SDP offer. Body: {sdp, type, session_id?}."""
    body = await request.json()
    base_url = f"{request.url.scheme}://{request.headers['host']}"
    session = store.get_or_create(body.get("session_id"), base_url)
    if old := peers.pop(session.id, None):
        await old.close()
    peer = Peer(session, handle, on_close=lambda p: _cleanup(p.session.id))
    peers[session.id] = peer
    answer = await peer.answer(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    return {"sdp": answer.sdp, "type": answer.type, "session_id": session.id}


@app.get("/assets/{session_id}/{name}")
async def asset(session_id: str, name: str) -> FileResponse:
    session = store.get(session_id)
    if session is None or "/" in name or name.startswith("."):
        raise HTTPException(404)
    path = session.dir / name
    if not path.is_file():
        raise HTTPException(404)
    return FileResponse(path)


def _cleanup(session_id: str) -> None:
    if w := watchers.pop(session_id, None):
        w.stop()
    peers.pop(session_id, None)


# --- message handler --------------------------------------------------------------


async def handle(peer: Peer, msg: m.Message) -> list[m.Message]:
    s = peer.session
    match msg:
        case m.Ping(t=t):
            return [m.Pong(t=t, server_t=int(time.time() * 1000))]
        case m.SessionStart():
            return [m.SessionState(session_id=s.id, task=s.task)]
        case m.TaskSet(text=text):
            s.task = text
            watcher = watchers.get(s.id) or Watcher(s, vlm, peer.send)
            watchers[s.id] = watcher
            watcher.start()
            return [
                m.SessionState(session_id=s.id, task=s.task),
                m.Status(phase="watching", text=text),
            ]
        case m.InputText(text=text):
            watcher = watchers.get(s.id)
            frames = watcher.recent_jpegs() if watcher else []
            if not frames and (latest := s.frames.latest()):
                frames = [_jpeg(latest)]
            peer.send(m.Status(phase="thinking"))
            t0 = time.monotonic()
            reply = await vlm.answer(text, frames, s.task)
            return [
                m.AgentMessage(
                    text=reply, status="answer", latency_ms=int((time.monotonic() - t0) * 1000)
                ),
                m.Status(phase="watching" if watcher and watcher.running else "idle"),
            ]
        case m.Capture(kind="photo", id=cid):
            frame = s.frames.latest()
            if frame is None:
                return [m.CaptureAck(id=cid, frames=0)]
            path = s.dir / f"capture_{len(s.captures):03d}.jpg"
            frame.to_jpeg(path)
            s.captures.append(path)
            return [m.CaptureAck(id=cid, frames=len(s.frames), url=s.asset_url(path))]
        case m.UiEvent(event=event, target=target):
            log.info("session %s: ui.event %s %s", s.id, event, target)
            return []
        case _:
            log.debug("session %s: unhandled %s", s.id, msg.type)
            return []


def _jpeg(frame: Frame) -> bytes:
    buf = io.BytesIO()
    frame.image.to_image().save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("aioice").setLevel(logging.WARNING)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
