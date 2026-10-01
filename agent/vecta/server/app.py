"""HTTP entrypoint: WebRTC signalling, assets, the message handler, and the
agent-facing endpoints the `vecta` CLI talks to.

Phone -> server events land in the session inbox (the agent tails it); the
agent answers through /sessions/{id}/say, /mark, /watch, /page, /send.
"""

from __future__ import annotations

import json
import logging
import os
import secrets
import time
from dataclasses import dataclass
from pathlib import Path

import uvicorn
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from vecta.protocol import messages as m
from vecta.server.audio import Listener, Transcriber
from vecta.server.ground import MarkerSpec, MarkerTracker
from vecta.server.keyframes import Keyframer
from vecta.server.rtc import Peer
from vecta.server.sessions import Session, SessionStore
from vecta.server.speech import Speaker
from vecta.server.vlm import Vlm
from vecta.server.watch import Watcher

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Settings:
    host: str = os.environ.get("VECTA_HOST", "0.0.0.0")
    port: int = int(os.environ.get("VECTA_PORT", "8000"))
    # the checkout's data/ (gitignored) unless overridden; this file is agent/vecta/server/app.py
    data_dir: Path = Path(
        os.environ.get("VECTA_DATA_DIR", Path(__file__).resolve().parents[3] / "data")
    )


settings = Settings()
app = FastAPI(title="vecta")
store = SessionStore(settings.data_dir)
vlm = Vlm()
stt = Transcriber()
tts = Speaker()


@dataclass
class Live:
    """Everything running for one connected phone."""

    peer: Peer
    keyframer: Keyframer
    markers: MarkerTracker
    watcher: Watcher | None = None


live: dict[str, Live] = {}


def _live(session_id: str) -> Live:
    if (lv := live.get(session_id)) is None:
        raise HTTPException(404, f"no live session {session_id}")
    return lv


# --- phone side ----------------------------------------------------------------------


@app.get("/ping")
async def ping() -> dict[str, int]:
    return {"t": int(time.time() * 1000)}


@app.post("/rtc/offer")
async def rtc_offer(request: Request) -> dict[str, str]:
    """Answer the phone's SDP offer. Body: {sdp, type, session_id?}."""
    body = await request.json()
    base_url = f"{request.url.scheme}://{request.headers['host']}"
    session = store.get_or_create(body.get("session_id"), base_url)
    if old := live.pop(session.id, None):
        await old.peer.close()
    peer = Peer(
        session,
        handle,
        on_close=lambda p: _on_peer_closed(p.session.id),
        on_audio=lambda track: _listen(session, peer, track),
    )
    lv = Live(
        peer=peer, keyframer=Keyframer(session), markers=MarkerTracker(session, vlm, peer.send)
    )
    live[session.id] = lv
    lv.keyframer.start()
    answer = await peer.answer(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    session.inbox.append("connected")
    return {"sdp": answer.sdp, "type": answer.type, "session_id": session.id}


async def _listen(session: Session, peer: Peer, track: MediaStreamTrack) -> None:
    def heard(text: str) -> None:
        session.inbox.append("voice", text=text)
        peer.send(m.Transcript(text=text))

    await Listener(stt, heard).run(track)


def _on_peer_closed(session_id: str) -> None:
    lv = live.pop(session_id, None)
    if lv is None:
        return
    lv.keyframer.stop()
    lv.markers.remove(None)
    if lv.watcher:
        lv.watcher.stop()
    lv.peer.session.inbox.append("disconnected")


@app.get("/assets/{session_id}/{path:path}")
async def asset(session_id: str, path: str) -> FileResponse:
    session = store.get(session_id)
    if session is None or ".." in path or path.startswith("/"):
        raise HTTPException(404)
    file = (session.dir / path).resolve()
    if not file.is_file() or session.dir.resolve() not in file.parents:
        raise HTTPException(404)
    return FileResponse(file)


async def handle(peer: Peer, msg: m.Message) -> list[m.Message]:
    s = peer.session
    match msg:
        case m.Ping(t=t):
            return [m.Pong(t=t, server_t=int(time.time() * 1000))]
        case m.SessionStart():
            return [m.SessionState(session_id=s.id, task=s.task)]
        case m.TaskSet(text=text):
            s.task = text
            s.inbox.append("task", text=text)
            return [m.SessionState(session_id=s.id, task=s.task)]
        case m.InputText(text=text):
            s.inbox.append("text", text=text)
            return []
        case m.Capture(kind="photo", id=cid):
            frame = s.frames.latest()
            if frame is None:
                return [m.CaptureAck(id=cid, frames=0)]
            path = s.dir / f"capture_{len(s.captures):03d}.jpg"
            frame.to_jpeg(path)
            s.captures.append(path)
            url = s.asset_url(path)
            s.inbox.append("capture", path=str(path), url=url)
            return [m.CaptureAck(id=cid, frames=len(s.frames), url=url)]
        case m.UiEvent(event=event, target=target, data=data):
            s.inbox.append("ui", event=event, target=target, data=data)
            return []
        case _:
            log.debug("session %s: unhandled %s", s.id, msg.type)
            return []


# --- agent side (the `vecta` CLI) -------------------------------------------------------


@app.get("/sessions")
async def sessions() -> list[dict]:
    out = []
    for sid, lv in live.items():
        s = lv.peer.session
        latest = s.frames.latest()
        out.append(
            {
                "session": sid,
                "connection": lv.peer.pc.connectionState,
                "task": s.task,
                "frames_received": s.frames.received,
                "latest_frame_age_ms": int(time.time() * 1000 - latest.ts_ms) if latest else None,
                "keyframes": lv.keyframer.count,
                "markers": list(lv.markers.markers),
                "watching": bool(lv.watcher and lv.watcher.running),
                "dir": str(s.dir),
            }
        )
    return out


@app.get("/sessions/{session_id}/inbox")
async def inbox(session_id: str, n: int = 50) -> list[dict]:
    session = store.get(session_id)
    if session is None:
        raise HTTPException(404)
    return session.inbox.tail(n)


@app.post("/sessions/{session_id}/say")
async def say(session_id: str, request: Request) -> dict:
    """Message the user; spoken by default."""
    body = await request.json()
    lv = _live(session_id)
    text, speak = body["text"], body.get("speak", True)
    lv.peer.send(m.AgentMessage(text=text, status=body.get("status", "answer")))
    lv.peer.session.inbox.append("agent", text=text, spoken=speak)
    if speak:
        try:
            for chunk in await tts.chunks(text):
                lv.peer.send(chunk)
        except Exception as e:
            log.warning("tts failed: %s", e)
            return {"sent": True, "spoken": False, "error": str(e)}
    return {"sent": True, "spoken": speak}


@app.post("/sessions/{session_id}/mark")
async def mark(session_id: str, request: Request) -> dict:
    """Pin a description to the live view until unmarked."""
    body = await request.json()
    lv = _live(session_id)
    spec = MarkerSpec(
        id=body.get("id") or secrets.token_urlsafe(4),
        query=body["query"],
        label=body.get("label", ""),
        color=body.get("color", "#46C46A"),
    )
    lv.markers.set(spec)
    return {"id": spec.id, "markers": list(lv.markers.markers)}


@app.post("/sessions/{session_id}/unmark")
async def unmark(session_id: str, request: Request) -> dict:
    body = await request.json()
    lv = _live(session_id)
    lv.markers.remove(body.get("id") or None)
    return {"markers": list(lv.markers.markers)}


@app.post("/sessions/{session_id}/watch")
async def watch(session_id: str, request: Request) -> dict:
    """Start (or, with empty text, stop) a VLM watch reporting to the inbox and the phone."""
    body = await request.json()
    lv = _live(session_id)
    s = lv.peer.session
    text = body.get("text", "").strip()
    if lv.watcher:
        lv.watcher.stop()
        lv.watcher = None
    if not text:
        lv.peer.send(m.Status(phase="idle"))
        return {"watching": False}
    s.task = text

    def report(msg: m.Message) -> None:
        if isinstance(msg, m.AgentMessage):
            s.inbox.append("watch", status=msg.status, text=msg.text, url=msg.url)
        lv.peer.send(msg)

    lv.watcher = Watcher(s, vlm, report)
    lv.watcher.start()
    lv.peer.send(m.Status(phase="watching", text=text))
    return {"watching": True, "text": text}


@app.post("/sessions/{session_id}/page")
async def page(session_id: str, request: Request) -> dict:
    body = await request.json()
    lv = _live(session_id)
    s = lv.peer.session
    lv.peer.send(m.PageRender(html=body["html"], version=s.next_page_version()))
    return {"version": s.page_version}


@app.post("/sessions/{session_id}/send")
async def send_raw(session_id: str, request: Request) -> dict:
    """Send any protocol message verbatim (escape hatch for the CLI)."""
    body = await request.json()
    lv = _live(session_id)
    try:
        msg = m.decode(json.dumps(body))
    except m.ProtocolError as e:
        raise HTTPException(400, str(e)) from e
    lv.peer.send(msg)
    return {"sent": msg.type}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    logging.getLogger("aioice").setLevel(logging.WARNING)
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
