"""HTTP entrypoint: WebRTC signalling, asset serving, and the v1 message handler.

The handler here is a stand-in for the agent: it acknowledges captures and
renders a demo page so the whole phone -> server -> page loop can be exercised
before any model is involved.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from html import escape
from pathlib import Path

import uvicorn
from aiortc import RTCSessionDescription
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from vecta.protocol import messages as m
from vecta.server.rtc import Peer
from vecta.server.sessions import Session, SessionStore

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("VECTA_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VECTA_PORT", "8000"))
    data_dir: Path = Path(os.environ.get("VECTA_DATA_DIR", Path.home() / ".vecta"))


settings = Settings()
app = FastAPI(title="vecta")
store = SessionStore(settings.data_dir)
peers: dict[str, Peer] = {}


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
    peer = Peer(session, handle)
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


# --- v1 stand-in handler -----------------------------------------------------------


async def handle(peer: Peer, msg: m.Message) -> list[m.Message]:
    s = peer.session
    match msg:
        case m.Ping(t=t):
            return [m.Pong(t=t, server_t=int(time.time() * 1000))]
        case m.SessionStart():
            return [m.SessionState(session_id=s.id, task=s.task)]
        case m.TaskSet(text=text) | m.InputText(text=text):
            s.task = text
            return [m.Status(phase="thinking"), render_demo(s), m.Status(phase="idle")]
        case m.Capture(kind="photo", id=cid):
            frame = s.frames.latest()
            if frame is None:
                return [m.CaptureAck(id=cid, frames=0)]
            path = s.dir / f"capture_{len(s.captures):03d}.jpg"
            frame.to_jpeg(path)
            s.captures.append(path)
            return [
                m.CaptureAck(id=cid, frames=len(s.frames), url=s.asset_url(path)),
                render_demo(s),
            ]
        case m.UiEvent(event=event, target=target):
            log.info("session %s: ui.event %s %s", s.id, event, target)
            return []
        case _:
            log.debug("session %s: unhandled %s", s.id, msg.type)
            return []


def render_demo(s: Session) -> m.PageRender:
    """Proof-of-loop page: task, stream stats, and the captures so far."""
    latest = s.frames.latest()
    size = f"{latest.size[0]}×{latest.size[1]}" if latest else "–"
    shots = "".join(
        f'<figure><img src="{s.asset_url(p)}" data-vecta-event="tap" data-vecta-id="{p.name}">'
        f"<figcaption>{p.name}</figcaption></figure>"
        for p in s.captures
    )
    html = f"""<!doctype html><meta name=viewport content="width=device-width,initial-scale=1">
<h1>{escape(s.task or "no task")}</h1>
<p>session <code>{s.id}</code> · {s.frames.received} frames received · {size}</p>
<div class=grid>{shots or "<p>tap the shutter to capture</p>"}</div>"""
    return m.PageRender(html=html, version=s.next_page_version())


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
