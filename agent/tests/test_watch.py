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
    assert kf.jpeg[:2] == b"\xff\xd8" and kf.jpeg_small[:2] == b"\xff\xd8"
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
    found = parse_verdict('{"status":"found","say":"mug on the left","count":2}')
    assert found == ("found", "mug on the left", 2)
    assert parse_verdict('{"status":"found","say":"a mug"}') == ("found", "a mug", 1)  # default
    assert parse_verdict('```json\n{"status": "searching", "say": ""}\n```') == ("searching", "", 0)
    assert parse_verdict("I can't tell") == ("info", "I can't tell", 0)
    assert parse_verdict('{"status":"weird"}') == ("info", '{"status":"weird"}', 0)


def test_found_reported_once_until_gone() -> None:
    """Only new instances reach the phone: found, found, found -> one message."""
    import asyncio

    from vecta.server.sessions import Session
    from vecta.server.vlm import Verdict
    from vecta.server.watch import Keyframe, Watcher

    class FakeVlm:
        def __init__(self, verdicts: list[Verdict]) -> None:
            self.verdicts = verdicts

        async def judge(self, task: str, jpeg: bytes) -> Verdict:
            return self.verdicts.pop(0)

    def v(status: str, count: int = 0) -> Verdict:
        return Verdict(status, "x" if status != "searching" else "", 1, count)

    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        session = Session(id="s", dir=Path(d), base_url="http://x", task="mug")
        sent: list = []
        script = [
            v("found", 1),
            v("found", 1),
            v("found", 2),
            v("searching"),
            v("searching"),
            v("found", 1),
        ]
        w = Watcher(session, FakeVlm(script), sent.append)  # type: ignore[arg-type]
        kf = Keyframe(FrameBuffer().push(frame(1)), b"\xff\xd8", b"\xff\xd8", 0)
        for _ in range(len(script)):
            asyncio.run(w._judge("mug", kf))
        # first sighting, a second mug arriving, and a fresh sighting after it left
        assert [(m.status, m.text) for m in sent] == [
            ("found", "x"),
            ("found", "x"),
            ("found", "x"),
        ]
        assert len(sent) == 3
