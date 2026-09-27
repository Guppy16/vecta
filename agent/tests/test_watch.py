import av
import numpy as np

from vecta.server.frames import FrameBuffer
from vecta.server.vlm import parse_verdict
from vecta.server.watch import _pick, dhash, hamming


def frame(seed: int, blur: bool = False) -> av.VideoFrame:
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
    if blur:
        img[:] = img.mean(axis=(0, 1)).astype(np.uint8)  # flat = zero sharpness
    return av.VideoFrame.from_ndarray(img, format="rgb24")


def test_pick_prefers_sharp_and_skips_duplicates() -> None:
    buf = FrameBuffer()
    buf.push(frame(1, blur=True))
    sharp = buf.push(frame(2))
    kf = _pick(buf, last_hash=None)
    assert kf is not None and kf.frame is sharp
    assert kf.jpeg[:2] == b"\xff\xd8"
    # same scene again -> nothing new to judge
    assert _pick(buf, last_hash=kf.hash) is None


def test_pick_empty() -> None:
    assert _pick(FrameBuffer(), None) is None


def test_dhash_distance() -> None:
    a = frame(1).to_ndarray(format="gray")
    b = frame(2).to_ndarray(format="gray")
    assert hamming(dhash(a), dhash(a)) == 0
    assert hamming(dhash(a), dhash(b)) > 8


def test_parse_verdict() -> None:
    assert parse_verdict('{"status":"found","say":"mug on the left"}') == (
        "found",
        "mug on the left",
    )
    assert parse_verdict('```json\n{"status": "searching", "say": ""}\n```') == ("searching", "")
    assert parse_verdict("I can't tell") == ("info", "I can't tell")
    assert parse_verdict('{"status":"weird"}') == ("info", '{"status":"weird"}')
