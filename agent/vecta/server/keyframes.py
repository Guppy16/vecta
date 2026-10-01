"""Record what the camera saw, task or no task.

A keyframe is a sharp frame that differs (dHash) from the last one kept. They
go to ``<session>/keyframes/`` at most once a second and are announced in the
inbox, so the agent gets a scan of a device without the user tapping the
shutter, and nothing seen during a session is lost.
"""

from __future__ import annotations

import asyncio
import logging
import time

from vecta.server.sessions import Session
from vecta.server.watch import _pick

log = logging.getLogger(__name__)

INTERVAL_S = 1.0


class Keyframer:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.dir = session.dir / "keyframes"
        self._last_hash: int | None = None
        self._task: asyncio.Task[None] | None = None
        self.count = 0

    def start(self) -> None:
        self.dir.mkdir(exist_ok=True)
        self._task = asyncio.create_task(self._run(), name=f"keyframes-{self.session.id}")

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while True:
            t0 = time.monotonic()
            kf = await asyncio.to_thread(_pick, self.session.frames, self._last_hash)
            if kf is not None:
                self._last_hash = kf.hash
                path = self.dir / f"kf_{self.count:04d}.jpg"
                path.write_bytes(kf.jpeg)
                self.count += 1
                self.session.inbox.append(
                    "keyframe", path=str(path), url=self.session.asset_url(path, sub="keyframes")
                )
            await asyncio.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))
