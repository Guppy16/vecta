import asyncio
import base64
import json
import struct
from pathlib import Path

import numpy as np
import pytest
from av import AudioResampler

from vecta.protocol import messages as m
from vecta.server.audio import FRAME_BYTES, FRAME_MS, Listener
from vecta.server.ground import parse_box
from vecta.server.inbox import Inbox
from vecta.server.sessions import SessionStore
from vecta.server.speech import (
    PhraseSplitter,
    _rate,
    _resample,
    _trim_silence,
    _wav_to_pcm16,
    earcon,
)
from vecta.server.talker import (
    HISTORY_TURNS,
    HOWTO_HINTS,
    LOOK_HINTS,
    ReplyExtractor,
    Talker,
    parse_reply,
)
from vecta.server.tts_track import SAMPLES, TtsTrack


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
    store = SessionStore(tmp_path)
    a = store.get_or_create("abc123-XYZ", "http://x")
    assert a.id == "abc123-XYZ" and a.dir.name == "abc123-XYZ"
    assert store.get_or_create("abc123-XYZ", "http://x") is a
    assert store.get_or_create("../evil", "http://x").id != "../evil"  # bad ids get a fresh one


def test_earcon_is_short_valid_pcm() -> None:
    chunk = earcon("heard")
    pcm = np.frombuffer(base64.b64decode(chunk.data), dtype=np.int16)
    assert 0.1 < len(pcm) / 16000 < 0.2 and abs(pcm).max() < 16000 and chunk.last


def test_talker_history_and_parse() -> None:
    assert parse_reply('{"reply": "Yes.", "escalate": false}') == ("Yes.", False, None)
    assert parse_reply('```json\n{"reply": "", "escalate": true}\n```') == ("", True, None)
    assert parse_reply('{"tool": "look"}') == ("", False, "look")
    assert parse_reply("plain text") == ("plain text", False, None)
    t = Talker()
    t.said("Hold it steady.")
    for i in range(HISTORY_TURNS + 5):
        t.heard(f"u{i}")
    assert len(t.history) == HISTORY_TURNS and t.history[-1]["content"] == f"u{HISTORY_TURNS + 4}"
    assert "BRIEFING" in t._system()


def test_tts_track_paces_and_fills_silence() -> None:
    async def main() -> None:
        t = TtsTrack()
        t.enqueue(b"\x01\x00" * (SAMPLES + 10))  # 1.03 frames of audio
        f1 = await t.recv()
        f2 = await t.recv()
        a1 = f1.to_ndarray().ravel()
        a2 = f2.to_ndarray().ravel()
        assert f1.sample_rate == 16000 and len(a1) == SAMPLES and (a1 == 1).all()
        assert (a2[:10] == 1).all() and (a2[10:] == 0).all()  # remainder, then silence
        assert f2.pts == SAMPLES

    asyncio.run(main())


def test_howto_and_look_hints() -> None:
    assert HOWTO_HINTS.search("how do I set the temperature")
    assert HOWTO_HINTS.search("what does this button do")
    assert HOWTO_HINTS.search("can you explain it")
    assert not HOWTO_HINTS.search("hello there")
    assert LOOK_HINTS.search("do you see the thermostat now?")
    assert not LOOK_HINTS.search("thanks")


def test_reply_extractor_streams_reply_text():
    raw = '{"reply": "It says \\"003\\", not a\\u00b0 temp.\\nOk", "escalate": false}'
    x, out = ReplyExtractor(), []
    for i in range(0, len(raw), 3):  # deltas cut anywhere, including inside escapes
        out.append(x.feed(raw[i : i + 3]))
    assert "".join(out) == 'It says "003", not a° temp.\nOk'
    assert x.done and x.started

    tool = ReplyExtractor()
    assert tool.feed('{"tool": "look"}') == "" and not tool.started


def test_phrase_splitter():
    p = PhraseSplitter()
    text = "The display shows zero zero three, which is a program number. Press PROG. Done"
    got = [ph for i in range(0, len(text), 4) for ph in p.feed(text[i : i + 4])] + p.flush()
    assert got == [
        "The display shows zero zero three,",  # first phrase may break at a clause
        "which is a program number.",
        "Press PROG.",
        "Done",
    ]
    short = PhraseSplitter()
    assert short.feed("Yes, I can hear you. ") == ["Yes,", "I can hear you."]


def test_resample_kokoro_stream():
    assert _rate("audio/l16;rate=24000;endianness=little-endian") == 24000
    r = AudioResampler(format="s16", layout="mono", rate=16000)
    pcm = (np.sin(np.arange(24000) / 10) * 8000).astype(np.int16).tobytes()  # 1 s at 24 kHz
    out = b"".join(_resample(r, pcm[i : i + 4800], 24000) for i in range(0, len(pcm), 4800))
    out += _resample(r, None, 24000)
    assert abs(len(out) // 2 - 16000) < 200


def test_trim_kokoro_padding():
    rate = 24000
    voice = (np.sin(np.arange(rate // 2) / 5) * 8000).astype(np.int16)  # 0.5 s
    pad = lambda s: np.zeros(int(rate * s), np.int16)  # noqa: E731
    out = _trim_silence(np.concatenate([pad(0.4), voice, pad(0.5)]).tobytes(), rate)
    assert abs(len(out) / 2 / rate - (0.03 + 0.5 + 0.15)) < 0.01
    assert _trim_silence(pad(0.2).tobytes(), rate) == b""


class _LoudVad:
    """Stand-in for webrtcvad: any non-zero frame is speech."""

    def is_speech(self, chunk: bytes, rate: int) -> bool:
        return any(chunk)


class _FakeStt:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, pcm: bytes) -> str:
        self.calls += 1
        return f"utterance of {len(pcm)} bytes"


def _frames(ms: int, loud: bool) -> bytes:
    return (b"\x01\x00" if loud else b"\x00\x00") * (FRAME_BYTES // 2) * (ms // FRAME_MS)


async def _feed(listener: Listener, pcm: bytes) -> None:
    listener._pending += pcm
    await listener._drain()
    await asyncio.sleep(0.01)  # let transcription tasks run


def test_listener_speculates_at_a_pause_and_commits_at_the_end():
    async def run() -> None:
        got: list[tuple[str, asyncio.Event]] = []

        def on_text(text, started, ended, committed):
            got.append((text, committed))
            return asyncio.create_task(committed.wait())

        listener = Listener(_FakeStt(), on_text)
        listener._vad = _LoudVad()
        await _feed(listener, _frames(600, True) + _frames(240, False))
        assert len(got) == 1 and not got[0][1].is_set()  # reply prepared, not yet heard
        await _feed(listener, _frames(150, False))
        assert got[0][1].is_set() and len(got) == 1  # end of utterance: same reply, now heard

    asyncio.run(run())


def test_listener_cancels_speculation_when_the_user_carries_on():
    async def run() -> None:
        replies: list[asyncio.Task] = []
        texts: list[str] = []

        def on_text(text, started, ended, committed):
            texts.append(text)
            replies.append(asyncio.create_task(committed.wait()))
            return replies[-1]

        listener = Listener(_FakeStt(), on_text)
        listener._vad = _LoudVad()
        await _feed(listener, _frames(600, True) + _frames(240, False))
        await _feed(listener, _frames(600, True) + _frames(390, False))
        assert replies[0].cancelled()
        assert len(texts) == 2 and texts[1] != texts[0]  # the second covers the whole utterance
        await asyncio.sleep(0)
        assert replies[1].done() and not replies[1].cancelled()

    asyncio.run(run())
