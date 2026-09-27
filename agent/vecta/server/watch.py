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
import json
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from vecta.protocol import messages as m
from vecta.server.frames import Frame, FrameBuffer
from vecta.server.sessions import Session
from vecta.server.vlm import Verdict, Vlm

log = logging.getLogger(__name__)

DEDUP_HAMMING = 8  # dHash distance below which two frames are "the same"
JUDGE_WIDTH = 448  # px sent to the VLM; ~3x faster than 720 with no loss on "is X there"
GONE_AFTER = 2  # consecutive "searching" verdicts before a found instance counts as gone
MIN_INTERVAL_S = 0.5  # floor between calls even if the model is fast
KEYFRAMES_KEPT = 6  # recent judged frames handed to `answer`


def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: 64 bits of 'is the pixel brighter than its left neighbour'."""
    small = np.asarray(Image.fromarray(gray).resize((size + 1, size), Image.Resampling.BILINEAR))
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def sharpness(gray: np.ndarray) -> float:
    """Variance of the 4-neighbour Laplacian; blur flattens it towards zero."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


@dataclass
class Keyframe:
    frame: Frame
    jpeg: bytes  # full resolution, for assets and the test set
    jpeg_small: bytes  # what the model sees
    hash: int


def _jpeg(img: Image.Image, quality: int = 85) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


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
    img = frame.image.to_image()
    small = img.resize(
        (JUDGE_WIDTH, int(img.height * JUDGE_WIDTH / img.width)), Image.Resampling.BILINEAR
    )
    return Keyframe(frame, _jpeg(img), _jpeg(small), h)


@dataclass
class Watcher:
    session: Session
    vlm: Vlm
    send: Callable[[m.Message], None]
    keyframes: list[Keyframe] = field(default_factory=list)
    _task: asyncio.Task[None] | None = None
    _last_hash: int | None = None
    _in_view: int = 0  # instances currently reported as found (0 = searching)
    _misses: int = 0  # consecutive "searching" verdicts while something was in view
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
        return [k.jpeg_small for k in self.keyframes[-KEYFRAMES_KEPT:]]

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

    def _record(self, kf: Keyframe, verdict: Verdict) -> None:
        """Keep every judged frame + verdict: a labelled test set falls out of normal use."""
        n = sum(1 for _ in self.session.dir.glob("judged_*.jpg"))
        path = self.session.dir / f"judged_{n:04d}.jpg"
        path.write_bytes(kf.jpeg)
        row = {
            "file": path.name,
            "task": self.session.task,
            "ts_ms": kf.frame.ts_ms,
            "status": verdict.status,
            "say": verdict.say,
            "latency_ms": verdict.latency_ms,
        }
        with (self.session.dir / "judged.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")

    async def _judge(self, task: str, kf: Keyframe) -> None:
        try:
            verdict = await self.vlm.judge(task, kf.jpeg_small)
        except Exception as e:  # keep watching through transient model errors
            log.warning("session %s: vlm error: %s", self.session.id, e)
            await asyncio.sleep(2)
            return
        self._last_hash = kf.hash
        self.last_verdict = f"{verdict.status} {verdict.latency_ms}ms {verdict.say}"
        self._record(kf, verdict)
        self.keyframes = (self.keyframes + [kf])[-KEYFRAMES_KEPT * 2 :]
        log.info(
            "session %s: %s %dms %r",
            self.session.id,
            verdict.status,
            verdict.latency_ms,
            verdict.say,
        )
        # Only new instances reach the user: a found is reported once, then stays quiet
        # while the model keeps confirming it, until it leaves view or the count grows.
        if verdict.status == "found":
            self._misses = 0
            if verdict.count <= self._in_view:
                return
            self._in_view = verdict.count
            path = self.session.dir / f"found_{int(time.time())}.jpg"
            path.write_bytes(kf.jpeg)
            url = self.session.asset_url(path)
        else:
            if self._in_view and verdict.status == "searching":
                self._misses += 1
                if self._misses >= GONE_AFTER:
                    self._in_view = 0
            if verdict.status != "info" or not verdict.say:
                return
            url = None
        self.send(
            m.AgentMessage(
                text=verdict.say, status=verdict.status, url=url, latency_ms=verdict.latency_ms
            )
        )
