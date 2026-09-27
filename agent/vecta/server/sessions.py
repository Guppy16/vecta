"""Per-phone session state: task, frame buffer, on-disk media, rendered pages."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from pathlib import Path

from vecta.server.frames import FrameBuffer


@dataclass
class Session:
    id: str
    dir: Path  # media and assets for this session live here
    base_url: str  # how the phone reaches this server, for asset URLs
    task: str | None = None
    frames: FrameBuffer = field(default_factory=FrameBuffer)
    page_version: int = 0
    captures: list[Path] = field(default_factory=list)

    def asset_url(self, path: Path) -> str:
        return f"{self.base_url}/assets/{self.id}/{path.name}"

    def next_page_version(self) -> int:
        self.page_version += 1
        return self.page_version


class SessionStore:
    def __init__(self, data_dir: Path) -> None:
        self._data_dir = data_dir
        self._sessions: dict[str, Session] = {}

    def create(self, base_url: str) -> Session:
        sid = secrets.token_urlsafe(8)
        session_dir = self._data_dir / "sessions" / sid
        session_dir.mkdir(parents=True, exist_ok=True)
        session = Session(id=sid, dir=session_dir, base_url=base_url)
        self._sessions[sid] = session
        return session

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def get_or_create(self, sid: str | None, base_url: str) -> Session:
        if sid and (existing := self.get(sid)):
            return existing
        return self.create(base_url)
