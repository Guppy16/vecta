import asyncio
import base64
import json
import struct
import time
from pathlib import Path

import numpy as np
import pytest
from av import AudioResampler
from fastapi import FastAPI
from fastapi.testclient import TestClient

from vecta.protocol import messages as m
from vecta.server import frame_labels, labels
from vecta.server.audio import FRAME_BYTES, FRAME_MS, Listener, Recorder
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
    HISTORY_KEEP,
    HISTORY_MAX,
    INSTRUCTION_HINTS,
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
    assert parse_reply('{"action": "answer", "reply": "Yes."}') == ("Yes.", False, None)
    assert parse_reply('```json\n{"action": "expert"}\n```') == ("", True, None)
    assert parse_reply('{"action": "look"}') == ("", False, "look")
    assert parse_reply('{"action": "silent"}') == ("", False, None)
    assert parse_reply("plain text") == ("plain text", False, None)
    t = Talker()
    t.said("Hold it steady.")
    t.said("", escalate=True)
    assert [json.loads(h["content"]) for h in t.history] == [
        {"action": "answer", "reply": "Hold it steady."},
        {"action": "expert"},
    ]
    system = t._system()
    t.brief("The thermostat is a Danfoss.")  # joins the dialogue; the system prompt is unchanged
    assert t._system() == system and "Danfoss" in t.history[-1]["content"]
    for i in range(HISTORY_MAX - 3):
        t.heard(f"u{i}")
    assert len(t.history) == HISTORY_MAX  # trimmed in one block, not message by message
    t.heard("one more")
    assert len(t.history) == HISTORY_KEEP and t.history[-1]["content"] == "one more"
    assert "Danfoss" in t._system()  # the trim folds the latest briefing into the prompt


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


def test_reply_extractor_streams_reply_text():
    raw = '{"action": "answer", "reply": "It says \\"003\\", not a\\u00b0 temp.\\nOk"}'
    x, out = ReplyExtractor(), []
    for i in range(0, len(raw), 3):  # deltas cut anywhere, including inside escapes
        out.append(x.feed(raw[i : i + 3]))
    assert "".join(out) == 'It says "003", not a° temp.\nOk'
    assert x.done and x.started

    tool = ReplyExtractor()
    assert tool.feed('{"action": "look"}') == "" and not tool.started


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


def _frames(ms: int, loud: bool) -> bytes:
    return (b"\x01\x00" if loud else b"\x00\x00") * (FRAME_BYTES // 2) * (ms // FRAME_MS)


async def _feed(listener: Listener, pcm: bytes) -> None:
    listener._pending += pcm
    await listener._drain()
    await asyncio.sleep(0.01)  # let the speech tasks run


def test_listener_speculates_at_a_pause_and_commits_at_the_end():
    async def run() -> None:
        got: list[tuple[bytes, asyncio.Event]] = []

        def on_speech(pcm, started, ended, committed):
            got.append((pcm, committed))
            return asyncio.create_task(committed.wait())

        listener = Listener(on_speech)
        listener._vad = _LoudVad()
        await _feed(listener, _frames(600, True) + _frames(240, False))
        assert len(got) == 1 and not got[0][1].is_set()  # reply prepared, not yet heard
        await _feed(listener, _frames(150, False))
        assert got[0][1].is_set() and len(got) == 1  # end of utterance: same reply, now heard

    asyncio.run(run())


def test_listener_cancels_speculation_when_the_user_carries_on():
    async def run() -> None:
        replies: list[asyncio.Task] = []
        heard: list[bytes] = []

        def on_speech(pcm, started, ended, committed):
            heard.append(pcm)
            replies.append(asyncio.create_task(committed.wait()))
            return replies[-1]

        listener = Listener(on_speech)
        listener._vad = _LoudVad()
        await _feed(listener, _frames(600, True) + _frames(240, False))
        await _feed(listener, _frames(600, True) + _frames(390, False))
        assert replies[0].cancelled()
        assert len(heard) == 2 and len(heard[1]) > len(heard[0])  # the whole utterance
        await asyncio.sleep(0)
        assert replies[1].done() and not replies[1].cancelled()

    asyncio.run(run())


def test_recorder_and_labels(tmp_path: Path):
    rec = Recorder(tmp_path / "sessions" / "abcdef1" / "utterances")
    uid = rec.save(b"\x00\x01" * 1600, started=1.0, ended=1.1, voiced_ms=100, kind="speech")
    rec.note(uid, text="hello")
    assert uid == "u_0001" and Recorder(rec.dir).save(b"") == "u_0002"  # numbering survives

    app = FastAPI()
    app.include_router(labels.router(tmp_path))
    client = TestClient(app)
    data = client.get("/label/items").json()
    first = next(u for u in data["items"] if u["id"] == "u_0001")
    assert first["hyps"][0]["text"] == "hello" and first["kind"] == "speech"
    assert first["duration_s"] == 0.1 and len(first["peaks"]) == 300
    assert client.get("/label/audio/abcdef1/u_0001.wav").status_code == 200
    assert client.get("/label/audio/abcdef1/..%2Fx.wav").status_code == 404
    body = {"session": "abcdef1", "id": "u_0001", "reference": "hello there"}
    assert client.post("/label", json=body).status_code == 200
    assert client.post("/label", json={"session": "abcdef1", "id": "u_0001"}).status_code == 400
    assert client.post("/label", json={**body, "noSpeech": True}).json()["reference"] == ""
    saved = client.get("/label/items").json()["labels"]
    assert saved["abcdef1/u_0001"]["noSpeech"] is True  # last save wins
    batch = [{"session": "abcdef1", "id": "u_0002", "exclude": True}]
    assert client.post("/label/batch", json={"items": batch}).json()[0]["exclude"] is True
    gone = client.post("/label/discard", json={"session": "abcdef1", "id": "u_0002"}).json()
    assert gone["discarded"] and not (rec.dir / "u_0002.wav").exists()
    assert all(u["id"] != "u_0002" for u in client.get("/label/items").json()["items"])
    rec.save(b"\x00\x00" * 160)  # u_0003
    batch = {"items": [{"session": "abcdef1", "id": "u_0003", "discard": True}]}
    assert client.post("/label/batch", json=batch).json()[0]["discarded"]
    assert not (rec.dir / "u_0003.wav").exists()
    bad = {"items": [{"session": "x", "id": "u_1"}]}
    assert client.post("/label/batch", json=bad).status_code == 400


def test_frame_labels(tmp_path: Path):
    sess = tmp_path / "sessions" / "abcdef1"
    sess.mkdir(parents=True)
    (sess / "judged_0000.jpg").write_bytes(b"\xff\xd8")
    judged = [
        {"file": "judged_0000.jpg", "task": "tell me when u see a tv", "status": "found"},
        # not an object the benchmark asks about
        {"file": "judged_0001.jpg", "task": "show me the boiler", "status": "found"},
    ]
    (sess / "judged.jsonl").write_text("".join(json.dumps(j) + "\n" for j in judged))
    (tmp_path / "calibration").mkdir()
    preds = {"exact": {"abcdef1/judged_0000.jpg": {"yes": False, "margin": -1.5}}}
    (tmp_path / "calibration" / "frame_preds.json").write_text(json.dumps(preds))

    app = FastAPI()
    app.include_router(frame_labels.router(tmp_path))
    client = TestClient(app)
    data = client.get("/label/frames/items").json()
    assert [f["key"] for f in data["items"]] == ["abcdef1/judged_0000.jpg"]
    frame = data["items"][0]
    assert frame["object"] == "tv" and frame["judged"]
    assert frame["answers"]["exact"]["margin"] == -1.5
    assert client.get("/label/frames/img/abcdef1/judged_0000.jpg").status_code == 200
    assert client.get("/label/frames/img/abcdef1/..%2Fx.jpg").status_code == 404
    body = {"key": "abcdef1/judged_0000.jpg", "present": "no", "note": "monitor"}
    assert client.post("/label/frames", json=body).status_code == 200
    assert client.post("/label/frames", json={**body, "present": "maybe"}).status_code == 422
    assert client.post("/label/frames", json={**body, "key": "../x"}).status_code == 422
    client.post("/label/frames", json={**body, "present": "unsure"})
    labels = client.get("/label/frames/items").json()["labels"]
    assert labels[body["key"]]["present"] == "unsure"  # the last save wins


def test_prune_keeps_only_labelled_recordings(tmp_path: Path):
    rec = Recorder(tmp_path / "sessions" / "abcdef1" / "utterances")
    ids = [rec.save(b"\x00\x00" * 160) for _ in range(4)]  # u_0001..u_0004
    blank = {"reference": "", "noSpeech": False, "unsure": False, "exclude": False, "note": ""}
    lines = [
        {**blank, "id": ids[0], "reference": "keep me"},
        {**blank, "id": ids[1], "exclude": True},
    ]
    with (tmp_path / "labels.jsonl").open("w") as f:
        for line in lines:
            f.write(json.dumps({"session": "abcdef1", **line}) + "\n")
    now = time.time()
    assert labels.prune(tmp_path, now) == []  # nothing is old yet
    gone = labels.prune(tmp_path, now + labels.KEEP_UNLABELLED_S + 1)
    assert sorted(p.stem for p in gone) == [ids[1], ids[2], ids[3]]
    assert (rec.dir / f"{ids[0]}.wav").exists()


def test_barge_in_stops_us_but_our_echo_does_not():
    async def run() -> None:
        heard: list[bytes] = []
        barged: list[bool] = []

        def on_speech(pcm, started, ended, committed):
            heard.append(pcm)
            return None

        listener = Listener(on_speech, on_barge_in=lambda: barged.append(True))
        listener._vad = _LoudVad()
        await listener.mute_for(5)  # we are talking
        await _feed(listener, _frames(300, True) + _frames(390, False))  # a short echo blip
        assert not barged and heard == []
        await _feed(listener, _frames(600, True) + _frames(390, False))  # the user cuts in
        assert barged == [True] and len(heard) == 1

    asyncio.run(run())


def test_instruction_hints_catch_improvised_steps():
    assert INSTRUCTION_HINTS.search("Step three is to press PROG, then use the minus button")
    assert INSTRUCTION_HINTS.search("Hold both arrows for three seconds.")
    assert not INSTRUCTION_HINTS.search("Yes, I can hear you. The display shows 22.5.")
    assert not INSTRUCTION_HINTS.search("I see a white plug with a black cable.")


def test_asks_closer_detects_a_request_for_a_close_up():
    from vecta.server.talker import ASKS_CLOSER

    assert ASKS_CLOSER.search("The brand isn't visible. Can you bring it closer?")
    assert ASKS_CLOSER.search("I can't read the label from here.")
    assert not ASKS_CLOSER.search("Now I can see it: the label says GA-52T.")


def test_instruction_guard_streams_until_the_reply_turns_into_steps():
    from vecta.server.app import _InstructionGuard

    class Voice:
        def __init__(self) -> None:
            self.text = ""

        def write(self, t: str) -> None:
            self.text += t

    v, reply = Voice(), "Let me check the label for you."
    g = _InstructionGuard(v)
    for i in range(0, len(reply), 3):
        g(reply[i : i + 3])
    assert not g.tripped and v.text == reply

    v = Voice()
    g = _InstructionGuard(v)
    g("Sure. First press PROG, then hold it.")
    assert g.tripped and v.text == ""
