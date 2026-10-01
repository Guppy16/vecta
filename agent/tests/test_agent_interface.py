import json
from pathlib import Path

import numpy as np
import pytest

from vecta.protocol import messages as m
from vecta.server.ground import parse_box
from vecta.server.inbox import Inbox
from vecta.server.speech import _wav_to_pcm16


def test_inbox_append_and_tail(tmp_path: Path) -> None:
    box = Inbox(tmp_path / "inbox.jsonl")
    assert box.tail() == []
    box.append("task", text="find the mug")
    box.append("voice", text="is this it?")
    events = box.tail()
    assert [e["type"] for e in events] == ["task", "voice"]
    assert events[1]["text"] == "is this it?" and "ts" in events[1]
    assert box.tail(1)[0]["type"] == "voice"
    # one JSON object per line, so `tail -f` works for an agent
    assert all(json.loads(line) for line in (tmp_path / "inbox.jsonl").read_text().splitlines())


def test_parse_box_qwen_grid() -> None:
    box = parse_box('{"found": true, "bbox_2d": [100, 200, 300, 400]}')
    assert box == pytest.approx((0.2, 0.3, 0.2, 0.2))
    assert parse_box('```json\n{"found": false}\n```') is None
    assert parse_box('{"bbox_2d": [500, 500, 400, 600]}') is None  # inverted box
    assert parse_box("no json here") is None


def test_wav_to_pcm16_resamples_float_wav() -> None:
    """A WAVE_FORMAT_IEEE_FLOAT file like Kokoro's (wave.open rejects these)."""
    import struct

    rate, seconds = 24000, 0.5
    t = np.arange(int(rate * seconds)) / rate
    data = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32).tobytes()
    fmt = struct.pack("<HHIIHH", 3, 1, rate, rate * 4, 4, 32)  # tag 3 = IEEE float
    wav = (
        b"RIFF"
        + struct.pack("<I", 4 + 8 + len(fmt) + 8 + len(data))
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", len(fmt))
        + fmt
        + b"data"
        + struct.pack("<I", len(data))
        + data
    )
    out = np.frombuffer(_wav_to_pcm16(wav), dtype=np.int16)
    assert abs(len(out) - 16000 * seconds) < 64  # resampler edge effects
    assert 0.4 < abs(out).max() / 32767 < 0.55


def test_new_message_types_round_trip() -> None:
    for msg in (
        m.OverlaySet(
            markers=[{"id": "a", "x": 0.5, "y": 0.5, "w": 0.1, "h": 0.1}], frame_w=720, frame_h=1280
        ),
        m.Transcript(text="hello", final=True),
        m.TtsChunk(id="t1", seq=0, last=True, data="AAAA"),
        m.Marker(id="a", label="1", x=0.2, y=0.3),
    ):
        assert m.decode(m.encode(msg)) == msg


def test_session_store_resumes_unknown_but_valid_id(tmp_path: Path) -> None:
    from vecta.server.sessions import SessionStore

    store = SessionStore(tmp_path)
    a = store.get_or_create("abc123-XYZ", "http://x")
    assert a.id == "abc123-XYZ" and a.dir.name == "abc123-XYZ"
    assert store.get_or_create("abc123-XYZ", "http://x") is a
    assert store.get_or_create("../evil", "http://x").id != "../evil"  # bad ids get a fresh one


def test_earcon_is_short_valid_pcm() -> None:
    import base64

    from vecta.server.speech import earcon

    chunk = earcon("heard")
    pcm = np.frombuffer(base64.b64decode(chunk.data), dtype=np.int16)
    assert 0.1 < len(pcm) / 16000 < 0.2 and abs(pcm).max() < 16000 and chunk.last
