"""Control-channel protocol between the phone and the server (v1).

Every message is one JSON object on the WebRTC data channel with a ``type``
field that selects the dataclass below. Media (video) travels on the WebRTC
video track, never here; large blobs the agent produces (stitched panels,
annotated frames) are served over HTTP and referenced from pages by URL.

The Kotlin side mirrors these shapes in `transport/Messages.kt`. Keep the two
in sync when changing anything here.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar

_REGISTRY: dict[str, type[Message]] = {}


@dataclass
class Message:
    """Base for all messages. Subclasses set ``type`` and register themselves."""

    type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        _REGISTRY[cls.type] = cls


# --- phone -> server -----------------------------------------------------------


@dataclass
class SessionStart(Message):
    """First message after the channel opens. ``session_id`` resumes a session."""

    type: ClassVar[str] = "session.start"
    session_id: str | None = None
    device: dict[str, Any] = field(default_factory=dict)  # {w, h, dpr}


@dataclass
class TaskSet(Message):
    """The user explicitly set or replaced the task."""

    type: ClassVar[str] = "task.set"
    text: str = ""


@dataclass
class InputText(Message):
    """Free text from the persistent input bar; the agent decides what it means."""

    type: ClassVar[str] = "input.text"
    text: str = ""


@dataclass
class UiEvent(Message):
    """A tap or form event from the rendered page's bridge."""

    type: ClassVar[str] = "ui.event"
    event: str = ""
    target: str | None = None
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class Capture(Message):
    """Marks the live stream: keep the current frame (photo) or a span (video)."""

    type: ClassVar[str] = "capture"
    kind: str = "photo"  # photo | video_start | video_stop
    id: str = ""
    ts: int = 0  # phone wall-clock ms, for pairing with the ack


@dataclass
class Ping(Message):
    type: ClassVar[str] = "ping"
    t: int = 0  # phone wall-clock ms


# --- server -> phone -----------------------------------------------------------


@dataclass
class SessionState(Message):
    type: ClassVar[str] = "session.state"
    session_id: str = ""
    task: str | None = None


@dataclass
class Status(Message):
    """Drives a small indicator on the phone; nothing else."""

    type: ClassVar[str] = "status"
    phase: str = "idle"  # idle | thinking | processing_media | awaiting_user
    text: str = ""


@dataclass
class PageRender(Message):
    """Full replacement of the webview contents."""

    type: ClassVar[str] = "page.render"
    html: str = ""
    version: int = 0


@dataclass
class CaptureRequest(Message):
    """The agent wants more input; ``hint`` is shown over the camera."""

    type: ClassVar[str] = "capture.request"
    kind: str = "photo"  # photo | video
    hint: str = ""


@dataclass
class CaptureAck(Message):
    """Confirms a ``Capture`` was recorded server-side."""

    type: ClassVar[str] = "capture.ack"
    id: str = ""
    frames: int = 0
    url: str | None = None  # where the captured still can be fetched


@dataclass
class Pong(Message):
    type: ClassVar[str] = "pong"
    t: int = 0  # echoed from the ping
    server_t: int = 0


# --- (de)serialisation ---------------------------------------------------------


class ProtocolError(ValueError):
    pass


def encode(msg: Message) -> str:
    return json.dumps({"type": msg.type, **asdict(msg)}, separators=(",", ":"))


def decode(text: str | bytes) -> Message:
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as e:
        raise ProtocolError(f"not JSON: {e}") from e
    if not isinstance(raw, dict) or "type" not in raw:
        raise ProtocolError("message must be an object with a 'type'")
    cls = _REGISTRY.get(raw["type"])
    if cls is None:
        raise ProtocolError(f"unknown message type {raw['type']!r}")
    known = {f.name for f in fields(cls)}
    # Ignore unknown keys so either side can add fields without breaking the other.
    return cls(**{k: v for k, v in raw.items() if k in known})
