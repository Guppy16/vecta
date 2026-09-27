from pathlib import Path

import av
import numpy as np

from vecta.server.frames import FrameBuffer


def fake_frame(w: int = 64, h: int = 48) -> av.VideoFrame:
    return av.VideoFrame.from_ndarray(np.zeros((h, w, 3), dtype=np.uint8), format="rgb24")


def test_ring_keeps_only_capacity() -> None:
    buf = FrameBuffer(capacity=3)
    for i in range(5):
        buf.push(fake_frame(), ts_ms=float(i))
    assert len(buf) == 3
    assert buf.received == 5
    assert buf.latest() is not None and buf.latest().ts_ms == 4.0
    assert [f.ts_ms for f in buf.since(3.0)] == [3.0, 4.0]


def test_empty_buffer() -> None:
    buf = FrameBuffer()
    assert buf.latest() is None
    assert buf.since(0) == []


def test_to_jpeg(tmp_path: Path) -> None:
    frame = FrameBuffer().push(fake_frame())
    out = tmp_path / "f.jpg"
    frame.to_jpeg(out)
    assert out.stat().st_size > 0
    assert frame.size == (64, 48)
