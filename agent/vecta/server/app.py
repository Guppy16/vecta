"""HTTP entrypoint: WebRTC signalling, assets, the message handler, and the
agent-facing endpoints the `vecta` CLI talks to.

Phone -> server events land in the session inbox (the agent tails it); the
agent answers through /sessions/{id}/say, /mark, /watch, /page, /send.
"""

import asyncio
import base64
import io
import json
import logging
import os
import secrets
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path

import uvicorn
from aiortc import RTCSessionDescription
from aiortc.mediastreams import MediaStreamTrack
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse

from vecta.protocol import messages as m
from vecta.server import frame_labels, labels
from vecta.server.audio import Listener, Recorder
from vecta.server.frames import Frame
from vecta.server.ground import MarkerSpec, MarkerTracker
from vecta.server.keyframes import Keyframer
from vecta.server.rtc import Peer
from vecta.server.sessions import Session, SessionStore
from vecta.server.speech import Speaker, SpeechStream, earcon
from vecta.server.talker import ASKS_CLOSER, INSTRUCTION_HINTS, Reply, Talker
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
    earcons: bool = os.environ.get("VECTA_EARCONS", "1") != "0"
    talker: bool = os.environ.get("VECTA_TALKER", "1") != "0"  # fast local voice replies


settings = Settings()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    task = asyncio.create_task(_prune_recordings(), name="prune-recordings")
    yield
    task.cancel()


async def _prune_recordings() -> None:
    """Hourly: unlabelled audio older than a day is deleted (labels.prune)."""
    while True:
        try:
            removed = await asyncio.to_thread(labels.prune, settings.data_dir)
            if removed:
                log.info("deleted %d unlabelled recordings older than a day", len(removed))
        except Exception:
            log.exception("pruning recordings failed")
        await asyncio.sleep(3600)


app = FastAPI(title="vecta", lifespan=lifespan)
app.include_router(labels.router(settings.data_dir))
app.include_router(frame_labels.router(settings.data_dir))
store = SessionStore(settings.data_dir)
vlm = Vlm()
tts = Speaker()


@dataclass
class Live:
    """Everything running for one connected phone."""

    peer: Peer
    keyframer: Keyframer
    markers: MarkerTracker
    watcher: Watcher | None = None
    listener: Listener | None = None
    talker: Talker = field(default_factory=Talker)
    voices: set[SpeechStream] = field(default_factory=set)  # speech in progress, for barge-in
    follow_up: asyncio.Task | None = None  # still looking after asking to bring it closer


live: dict[str, Live] = {}
talkers: dict[str, Talker] = {}  # outlive connections; see _talker


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
        old.keyframer.stop()
        old.markers.remove(None)
        if old.watcher:
            old.watcher.stop()
        await old.peer.close()
    peer = Peer(
        session,
        handle,
        on_close=_on_peer_closed,
        on_audio=lambda track: _listen(session, peer, track),
    )
    lv = Live(
        peer=peer,
        keyframer=Keyframer(session),
        markers=MarkerTracker(session, vlm, peer.send),
        talker=_talker(session),
    )
    live[session.id] = lv
    lv.keyframer.start()
    answer = await peer.answer(RTCSessionDescription(sdp=body["sdp"], type=body["type"]))
    session.inbox.append("connected")
    return {"sdp": answer.sdp, "type": answer.type, "session_id": session.id}


def _talker(session: Session) -> Talker:
    """One talker per session for the life of the server, so a reconnect keeps the
    conversation; after a restart it starts from the last briefing in the inbox."""
    if session.id not in talkers:
        briefs = [e for e in session.inbox.tail(1_000_000) if e.get("type") == "brief"]
        talkers[session.id] = Talker(briefing=briefs[-1]["text"]) if briefs else Talker()
    return talkers[session.id]


async def _listen(session: Session, peer: Peer, track: MediaStreamTrack) -> None:
    def on_speech(pcm: bytes, started: float, ended: float, committed: asyncio.Event):
        return asyncio.create_task(_on_voice(session, peer, pcm, started, ended, committed))

    def barge_in() -> None:
        if lv := live.get(session.id):
            _stop_talking(lv)

    listener = Listener(on_speech, Recorder(session.dir / "utterances"), barge_in)
    if lv := live.get(session.id):
        lv.listener = listener
    await listener.run(track)


def _stop_talking(lv: Live) -> None:
    """The user interrupted: drop what we were about to say and what is still queued."""
    for voice in list(lv.voices):
        voice.cancel()
    lv.peer.voice.clear()
    lv.peer.session.inbox.append("barge_in")


async def _on_voice(
    session: Session,
    peer: Peer,
    pcm: bytes,
    started: float,
    ended: float,
    committed: asyncio.Event,
) -> str:
    """One utterance. The talker hears it and starts deciding straight away while the same
    audio is transcribed; everything the user or the inbox can see waits for `committed`
    (the listener may still decide it was only a pause and cancel this task). Returns the
    transcript."""
    lv = live.get(session.id)
    if lv is None:
        return ""
    if lv.follow_up:
        lv.follow_up.cancel()  # they are talking again: stop waiting for the close-up
    saved = list(lv.talker.history)
    heard = lv.talker.hear(pcm)
    talk = asyncio.create_task(_talk(lv, heard, ended, committed)) if settings.talker else None
    try:
        await committed.wait()
    except asyncio.CancelledError:
        heard.cancel()
        if talk:
            talk.cancel()
        lv.talker.history[:] = saved  # it was only a pause; the user is still talking
        raise
    if settings.earcons:
        peer.voice.enqueue(base64.b64decode(earcon("heard").data))  # instant "got that"
    text = await heard
    if text:
        session.inbox.append("voice", text=text, started_at=round(started, 3))
        peer.send(m.Transcript(text=text))
    if talk:
        await talk
    return text


async def _talk(lv: Live, heard: asyncio.Task[str], ended_at: float, committed: asyncio.Event):
    """Fast loop: the talker answers (or stays quiet) and its reply is spoken while it is
    still being written; escalations go to the main agent's inbox.

    Runs speculatively: nothing is sent, logged or played until `committed` is set; if the
    task is cancelled, _on_voice makes the talker forget the turn. `heard` gives the
    transcript; `ended_at` is when the user stopped speaking (epoch s), for the
    voice-to-voice latency.
    """
    s = lv.peer.session

    async def sink(pcm: bytes) -> None:
        await committed.wait()
        await _play(lv, pcm)

    voice = SpeechStream(tts, sink, lv.voices)
    guard = _InstructionGuard(voice)
    try:
        reply = await lv.talker.turn(on_text=guard)
        if reply.tool == "look":
            reply = await _look_and_answer(lv, heard, guard, committed)
        if guard.tripped:  # it started explaining how to operate something: that's ours
            voice.drop_unsaid()
            reply = Reply(guard.said.strip(), True, reply.latency_ms)
            lv.talker.said("", escalate=True)
        await committed.wait()
    except asyncio.CancelledError:
        voice.cancel()
        raise
    except Exception as e:
        log.warning("talker failed: %s", e)
        voice.cancel()
        return
    if reply.escalate:
        s.inbox.append("escalate", text=await heard, context=_context(lv.talker))
        lv.peer.send(m.Status(phase="thinking"))
        if not reply.text:  # handed over: a short chime says "working on it"
            lv.peer.voice.enqueue(base64.b64decode(earcon("thinking").data))
    if reply.text:
        lv.peer.send(m.AgentMessage(text=reply.text, status="answer", latency_ms=reply.latency_ms))
    await voice.finish()
    if reply.text:
        v2v = _since(ended_at, voice.first_audio_at)
        log.info("talker %r: first text %s ms, v2v %s ms", reply.text, reply.first_text_ms, v2v)
        s.inbox.append(
            "agent",
            text=reply.text,
            by="talker",
            latency_ms=reply.latency_ms,
            voice_to_voice_ms=v2v,
        )


def _context(talker: Talker) -> list[dict]:
    """The last few turns for the inbox, with audio not yet transcribed left out."""
    return [
        {**msg, "content": msg["content"] if isinstance(msg["content"], str) else "[audio]"}
        for msg in talker.history[-6:]
    ]


class _InstructionGuard:
    """Sits between the talker's streaming text and the voice: once the reply turns into
    instructions (press this, hold that), nothing more of it is spoken."""

    def __init__(self, voice: SpeechStream) -> None:
        self._voice = voice
        self._text = ""
        self.said = ""  # the part let through before it tripped
        self.tripped = False

    def __call__(self, delta: str) -> None:
        if self.tripped:
            return
        self._text += delta
        if INSTRUCTION_HINTS.search(self._text):
            self.tripped = True
            return
        self._voice.write(self._text[len(self.said) :])
        self.said = self._text


async def _look_and_answer(
    lv: Live, heard: asyncio.Task[str], on_text, committed: asyncio.Event
) -> Reply:
    """The talker's one tool: it looks at the current frame itself (Nemotron is multimodal)."""
    s = lv.peer.session
    frame = s.frames.latest()
    if frame is None:
        reply = await lv.talker.tool_result("look", "No camera frame has arrived yet.", on_text)
        await committed.wait()
        return reply
    jpeg = await asyncio.to_thread(_jpeg_small, frame)
    if await asyncio.to_thread(_brightness, jpeg) < 12:
        reply = await lv.talker.tool_result("look", BLACK_FRAME, on_text)
    else:
        reply = await lv.talker.look_result(jpeg, on_text)
    await committed.wait()
    # fast answer, slow verification: the main agent fact-checks every vision answer
    # against the frame the talker saw and corrects out loud if it got it wrong
    looks = s.dir / "looks"
    looks.mkdir(exist_ok=True)
    seen = looks / f"look_{int(time.time() * 1000)}.jpg"
    seen.write_bytes(jpeg)
    question = await heard
    s.inbox.append("verify", question=question, answer=reply.text, frame=str(seen))
    if reply.text and ASKS_CLOSER.search(reply.text):
        lv.follow_up = asyncio.create_task(_follow_up(lv, question))
    return reply


FOLLOW_UP_EVERY_S, FOLLOW_UP_FOR_S = 1.5, 12.0


async def _follow_up(lv: Live, question: str) -> None:
    """The talker asked the user to bring something closer: keep looking for a while and say
    the answer as soon as it can tell, without being asked again. Cancelled when the user
    speaks (see _on_voice)."""
    s = lv.peer.session
    deadline = time.monotonic() + FOLLOW_UP_FOR_S
    while time.monotonic() < deadline:
        await asyncio.sleep(FOLLOW_UP_EVERY_S)
        frame = s.frames.latest()
        if frame is None:
            continue
        jpeg = await asyncio.to_thread(_jpeg_small, frame)
        try:
            answer = await lv.talker.recheck(question, jpeg)
        except Exception as e:
            log.warning("follow-up look failed: %s", e)
            return
        if answer:
            lv.talker.said(answer)
            seen = s.dir / "looks" / f"look_{int(time.time() * 1000)}.jpg"
            seen.write_bytes(jpeg)
            s.inbox.append("verify", question=question, answer=answer, frame=str(seen))
            s.inbox.append("agent", text=answer, by="talker", follow_up=True)
            await _speak(lv, answer)
            return


def _since(epoch_s: float | None, mono_s: float | None) -> int | None:
    """ms from an epoch timestamp to a monotonic one (both taken on this machine)."""
    if epoch_s is None or mono_s is None:
        return None
    return int((mono_s - time.monotonic() + time.time() - epoch_s) * 1000)


BLACK_FRAME = "The camera image is completely black: the lens is covered or the phone is face down."


async def describe_view(s: Session) -> str:
    """What the camera shows right now, in a sentence — for the talker and `vecta look`."""
    frame = s.frames.latest()
    if frame is None:
        return "No camera frame has arrived yet."
    jpeg = await asyncio.to_thread(_jpeg_small, frame)
    brightness = await asyncio.to_thread(_brightness, jpeg)
    if brightness < 12:
        return BLACK_FRAME
    text = await vlm.answer(
        "Describe what is in view in one or two short sentences, naming any device, "
        "buttons, labels or display text you can read.",
        [jpeg],
        s.task,
    )
    return text


def _jpeg_small(frame: Frame) -> bytes:
    img = frame.image.to_image()
    img = img.resize((448, int(img.height * 448 / img.width)))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _brightness(jpeg: bytes) -> float:
    from PIL import Image, ImageStat

    return ImageStat.Stat(Image.open(io.BytesIO(jpeg)).convert("L")).mean[0]


async def _speak(
    lv: Live,
    text: str,
    status: str = "answer",
    latency_ms: int | None = None,
) -> bool:
    lv.peer.send(m.AgentMessage(text=text, status=status, latency_ms=latency_ms))
    voice = SpeechStream(tts, lambda pcm: _play(lv, pcm), lv.voices)
    voice.write(text)
    await voice.finish()
    return voice.first_audio_at is not None


async def _play(lv: Live, pcm: bytes) -> None:
    # WebRTC's echo canceller handles most of our voice; the mute covers what it doesn't
    seconds = lv.peer.voice.pending_seconds + len(pcm) / (16000 * 2)
    if lv.listener:
        await lv.listener.mute_for(seconds + 0.5)
    lv.peer.voice.enqueue(pcm)


def _on_peer_closed(peer: Peer) -> None:
    lv = live.get(peer.session.id)
    if lv is None or lv.peer is not peer:
        return  # a newer connection already replaced this one
    live.pop(peer.session.id)
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
    """The main agent messages the user; spoken by default. The talker hears it too."""
    body = await request.json()
    lv = _live(session_id)
    text, speak = body["text"], body.get("speak", True)
    lv.peer.session.inbox.append("agent", text=text, by="main", spoken=speak)
    lv.talker.said(text)
    if not speak:
        lv.peer.send(m.AgentMessage(text=text, status=body.get("status", "answer")))
        return {"sent": True, "spoken": False}
    spoken = await _speak(lv, text, status=body.get("status", "answer"))
    lv.peer.send(m.Status(phase="idle"))
    return {"sent": True, "spoken": spoken}


@app.get("/sessions/{session_id}/look")
async def look(session_id: str) -> dict:
    """Latest keyframe + a one-line description, for the main agent."""
    lv = _live(session_id)
    s = lv.peer.session
    frames = (
        sorted((s.dir / "keyframes").glob("kf_*.jpg")) if (s.dir / "keyframes").is_dir() else []
    )
    latest = frames[-1] if frames else None
    return {
        "description": await describe_view(s),
        "keyframe": str(latest) if latest else None,
        "url": s.asset_url(latest, sub="keyframes") if latest else None,
        "frames_received": s.frames.received,
    }


@app.post("/sessions/{session_id}/brief")
async def brief(session_id: str, request: Request) -> dict:
    """Update what the talker knows (the main agent's current understanding and intent)."""
    body = await request.json()
    lv = _live(session_id)
    lv.talker.brief(body["text"])
    lv.peer.session.inbox.append("brief", text=body["text"])
    return {"briefing": lv.talker.briefing}


@app.post("/sessions/{session_id}/task")
async def set_task(session_id: str, request: Request) -> dict:
    """Set the task label the phone shows (the agent's understanding of the job)."""
    body = await request.json()
    lv = _live(session_id)
    s = lv.peer.session
    s.task = body.get("text") or None
    s.inbox.append("task", text=s.task, by="agent")
    lv.peer.send(m.SessionState(session_id=s.id, task=s.task))
    return {"task": s.task}


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
