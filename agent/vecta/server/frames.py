"""Rolling buffer of decoded video frames from one phone.

Frames arrive already decoded from aiortc (``av.VideoFrame``). We keep the last
few seconds in memory so a capture can be answered from history ("keep what was
on screen when the user tapped") and so later tiers can pull short clips
without asking the user to rescan.
"""

import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import av


@dataclass(frozen=True)
class Frame:
    ts_ms: float  # server monotonic-ish wall clock at receive
    image: av.VideoFrame

    @property
    def size(self) -> tuple[int, int]:
        return self.image.width, self.image.height

    def to_jpeg(self, path: Path, quality: int = 92) -> None:
        self.image.to_image().save(path, format="JPEG", quality=quality)


class FrameBuffer:
    """Fixed-capacity ring of recent frames. Not thread-safe; asyncio-only."""

    def __init__(self, capacity: int = 90) -> None:
        self._frames: deque[Frame] = deque(maxlen=capacity)
        self.received = 0

    def push(self, image: av.VideoFrame, ts_ms: float | None = None) -> Frame:
        frame = Frame(ts_ms if ts_ms is not None else time.time() * 1000, image)
        self._frames.append(frame)
        self.received += 1
        return frame

    def latest(self) -> Frame | None:
        return self._frames[-1] if self._frames else None

    def since(self, ts_ms: float) -> list[Frame]:
        return [f for f in self._frames if f.ts_ms >= ts_ms]

    def __len__(self) -> int:
        return len(self._frames)
