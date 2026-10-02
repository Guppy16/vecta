"""Per-phone session state: task, frame buffer, on-disk media, rendered pages."""

import re
import secrets
from dataclasses import dataclass, field
from pathlib import Path

from vecta.server.frames import FrameBuffer
from vecta.server.inbox import Inbox

_SAFE_ID = re.compile(r"[A-Za-z0-9_-]{6,32}")


@dataclass
class Session:
    id: str
    dir: Path  # media and assets for this session live here
    base_url: str  # how the phone reaches this server, for asset URLs
    task: str | None = None
    frames: FrameBuffer = field(default_factory=FrameBuffer)
    page_version: int = 0
    captures: list[Path] = field(default_factory=list)
    inbox: Inbox = field(init=False)

    def __post_init__(self) -> None:
        self.inbox = Inbox(self.dir / "inbox.jsonl")

    def asset_url(self, path: Path, sub: str = "") -> str:
        rel = f"{sub}/{path.name}" if sub else path.name
        return f"{self.base_url}/assets/{self.id}/{rel}"

    def next_page_version(self) -> int:
        self.page_version += 1
        return self.page_version


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._sessions: dict[str, Session] = {}

    def create(self, base_url: str, sid: str | None = None) -> Session:
        sid = sid or secrets.token_urlsafe(8)
        session_dir = self._data_dir / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        session = Session(id=sid, dir=session_dir, base_url=base_url)
        self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def get_or_create(self, sid: str | None, base_url: str) -> Session:
        """Resume by id; an unknown but well-formed id (e.g. after a server restart)
        is re-created in place so the phone keeps its identity and its folder."""
        if sid and (existing := self.get(sid)):
            return existing
        if sid and _SAFE_ID.fullmatch(sid):
            return self.create(base_url, sid)
        return self.create(base_url)
