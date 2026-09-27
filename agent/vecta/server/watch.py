"""The standing watch: keep asking the VLM whether the task is met in what the
camera sees, without sending it every frame.

Selection rules (carried over from the first prototype):
- one VLM call in flight at a time; frames keep arriving meanwhile
- pick the sharpest recent frame that is visually distinct (dHash) from the
  last one judged, so a static scene costs almost nothing and a pan sends
  fresh views
- don't repeat "found" while the thing stays in view
"""

from __future__ import annotations

import asyncio
import io
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import cv2
import numpy as np

from vecta.protocol import messages as m
from vecta.server.frames import Frame, FrameBuffer
from vecta.server.sessions import Session
from vecta.server.vlm import Vlm

log = logging.getLogger(__name__)

DEDUP_HAMMING = 8  # dHash distance below which two frames are "the same"
FOUND_DEBOUNCE_S = 6.0
MIN_INTERVAL_S = 0.5  # floor between calls even if the model is fast
KEYFRAMES_KEPT = 6  # recent judged frames handed to `answer`


def dhash(gray: np.ndarray, size: int = 8) -> int:
    small = cv2.resize(gray, (size + 1, size))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class Keyframe:
    frame: Frame
    jpeg: bytes
    hash: int


PICK_STRIDE = 5  # judge sharpness on every Nth recent frame; Laplacian on 720p isn't free


def _pick(buffer: FrameBuffer, last_hash: int | None) -> Keyframe | None:
    """Sharpest frame in the last second that differs from `last_hash`, else None.

    CPU-bound (decode to gray + Laplacian + JPEG); call via `asyncio.to_thread`.
    """
    cutoff = time.time() * 1000 - 1000
    best: tuple[float, Frame, np.ndarray] | None = None
    for f in buffer.since(cutoff)[::-PICK_STRIDE]:
        gray = f.image.to_ndarray(format="gray")
        s = sharpness(gray)
        if best is None or s > best[0]:
            best = (s, f, gray)
    if best is None:
        return None
    _, frame, gray = best
    h = dhash(gray)
    if last_hash is not None and hamming(h, last_hash) < DEDUP_HAMMING:
        return None
    buf = io.BytesIO()
    frame.image.to_image().save(buf, format="JPEG", quality=85)
    return Keyframe(frame, buf.getvalue(), h)


@dataclass
class Watcher:
    session: Session
    vlm: Vlm
    send: Callable[[m.Message], None]
    keyframes: list[Keyframe] = field(default_factory=list)
    _task: asyncio.Task[None] | None = None
    _last_hash: int | None = None
    _last_found: float = 0.0
    last_pick_at: float = 0.0
    last_verdict: str = ""

    def start(self) -> None:
        self.stop()
        self._task = asyncio.create_task(self._run(), name=f"watch-{self.session.id}")
        self._task.add_done_callback(self._report)

    def _report(self, task: asyncio.Task[None]) -> None:
        if not task.cancelled() and task.exception() is not None:
            log.error("session %s: watch task died", self.session.id, exc_info=task.exception())

    def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            self._task = None

    @property
    def running(self) -> bool:
        return self._task is not None and not self._task.done()

    def recent_jpegs(self) -> list[bytes]:
        return [k.jpeg for k in self.keyframes[-KEYFRAMES_KEPT:]]

    async def _run(self) -> None:
        task = self.session.task or ""
        log.info("session %s: watching for %r", self.session.id, task)
        while True:
            t0 = time.monotonic()
            kf = await asyncio.to_thread(_pick, self.session.frames, self._last_hash)
            if kf is not None:
                await self._judge(task, kf)
            self.last_pick_at = time.time()
            await asyncio.sleep(max(0.0, MIN_INTERVAL_S - (time.monotonic() - t0)))

    async def _judge(self, task: str, kf: Keyframe) -> None:
        try:
            verdict = await self.vlm.judge(task, kf.jpeg)
        except Exception as e:  # keep watching through transient model errors
            log.warning("session %s: vlm error: %s", self.session.id, e)
            await asyncio.sleep(2)
            return
        self._last_hash = kf.hash
        self.last_verdict = f"{verdict.status} {verdict.latency_ms}ms {verdict.say}"
        self.keyframes = (self.keyframes + [kf])[-KEYFRAMES_KEPT * 2 :]
        log.info(
            "session %s: %s %dms %r",
            self.session.id,
            verdict.status,
            verdict.latency_ms,
            verdict.say,
        )
        if verdict.status == "found":
            now = time.monotonic()
            if now - self._last_found < FOUND_DEBOUNCE_S:
                return
            self._last_found = now
            path = self.session.dir / f"found_{int(time.time())}.jpg"
            path.write_bytes(kf.jpeg)
            url = self.session.asset_url(path)
        elif verdict.status == "info" and verdict.say:
            url = None
        else:
            return  # searching: nothing to say
        self.send(
            m.AgentMessage(
                text=verdict.say, status=verdict.status, url=url, latency_ms=verdict.latency_ms
            )
        )
