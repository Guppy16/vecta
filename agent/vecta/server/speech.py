"""Agent speech: text -> Kokoro (via Lemonade) -> 16 kHz PCM16 for the phone's audio track.

Replies are spoken while they are still being written: the talker's text is cut into
phrases (PhraseSplitter), each phrase is synthesised with Kokoro's streaming mode, and
audio goes out as soon as it exists (SpeechStream).
"""

import asyncio
import base64
import io
import logging
import os
import re
import time
from collections.abc import Awaitable, Callable

import av
import httpx
import numpy as np
from av import AudioFrame, AudioResampler

from vecta.protocol import messages as m

log = logging.getLogger(__name__)

OUT_RATE = 16000
KOKORO_RATE = 24000  # Kokoro's native rate; its stream says so in the content type too
SILENCE = 500  # |PCM16 sample| below this (~ -36 dBFS) counts as silence
KEEP_BEFORE_S, KEEP_AFTER_S = 0.03, 0.15  # natural onset, and a short pause between phrases


class Speaker:
    def __init__(
        self,
        base_url: str = os.environ.get("VECTA_LLM_BASE_URL", "http://127.0.0.1:13305/api/v1"),
        model: str = os.environ.get("VECTA_TTS_MODEL", "kokoro-v1"),
        voice: str = os.environ.get("VECTA_TTS_VOICE", "af_heart"),
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60)
        self.model, self.voice = model, voice

    async def synthesize(self, text: str) -> bytes:
        """PCM16 mono at OUT_RATE."""
        r = await self._client.post(
            "/audio/speech",
            json={
                "model": self.model,
                "input": text,
                "voice": self.voice,
                "response_format": "wav",
            },
        )
        r.raise_for_status()
        return _wav_to_pcm16(r.content)

    async def phrase(self, text: str) -> bytes:
        """One short phrase as PCM16 mono at OUT_RATE, with Kokoro's padding trimmed.

        Kokoro's stream mode returns raw PCM but still synthesises a sentence at a time
        (~150 ms + ~3 ms per character), so phrases are kept short instead. It pads every
        request with ~0.35 s of silence before and ~0.5 s after, which would be dead air
        before the first word and between phrases.
        """
        body = {"model": self.model, "input": text, "voice": self.voice, "stream": True}
        r = await self._client.post("/audio/speech", json=body)
        r.raise_for_status()
        rate = _rate(r.headers.get("content-type", ""))
        pcm = _trim_silence(r.content[: len(r.content) // 2 * 2], rate)
        resampler = AudioResampler(format="s16", layout="mono", rate=OUT_RATE)
        return _resample(resampler, pcm, rate) + _resample(resampler, None, rate)


SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s+")
CLAUSE_END = re.compile(r"[,;:—]\s+")


class PhraseSplitter:
    """Cuts streaming text into phrases to synthesise one by one: at sentence ends, and for
    the first phrase also at a clause break ("Yes," on its own), so the first audio starts as
    early as possible; the next phrase is synthesised while that one plays."""

    def __init__(self) -> None:
        self._buf = ""
        self._first = True

    def feed(self, text: str) -> list[str]:
        self._buf += text
        out = []
        while True:
            cut = SENTENCE_END.search(self._buf)
            if self._first:
                clause = CLAUSE_END.search(self._buf)
                if clause and (cut is None or clause.start() < cut.start()):
                    cut = clause
            if cut is None:
                return out
            phrase, self._buf = self._buf[: cut.end()].strip(), self._buf[cut.end() :]
            if phrase:
                out.append(phrase)
                self._first = False

    def flush(self) -> list[str]:
        phrase, self._buf = self._buf.strip(), ""
        return [phrase] if phrase else []


class SpeechStream:
    """One spoken reply, fed phrase by phrase while it is still being written.

    Phrases are synthesised in order by a single worker and each goes to `sink` as soon as
    it is ready. Synthesis runs faster than real time, so the next phrase is ready before
    the current one finishes playing.
    """

    def __init__(
        self,
        speaker: Speaker,
        sink: Callable[[bytes], Awaitable[None]],
        active: set[SpeechStream] | None = None,
    ) -> None:
        """`active`: a set this stream sits in while it speaks, so it can be stopped (barge-in)."""
        self._speaker, self._sink = speaker, sink
        self._active = active if active is not None else set()
        self._active.add(self)
        self._queue: asyncio.Queue[str | None] = asyncio.Queue()
        self._splitter = PhraseSplitter()
        self.first_audio_at: float | None = None  # time.monotonic()
        self._task = asyncio.create_task(self._run())

    def write(self, text: str) -> None:
        """Streaming text in; complete phrases are queued for synthesis."""
        for phrase in self._splitter.feed(text):
            self._queue.put_nowait(phrase)

    async def finish(self) -> None:
        """No more text: speak the remainder and wait until all audio has been handed over."""
        for phrase in self._splitter.flush():
            self._queue.put_nowait(phrase)
        self._queue.put_nowait(None)
        await asyncio.wait([self._task])  # returns normally even if the stream was cancelled

    def cancel(self) -> None:
        self._task.cancel()

    async def _run(self) -> None:
        try:
            await self._speak_queued()
        finally:
            self._active.discard(self)

    async def _speak_queued(self) -> None:
        while (phrase := await self._queue.get()) is not None:
            try:
                pcm = await self._speaker.phrase(phrase)
            except Exception as e:
                log.warning("tts failed for %r: %s", phrase, e)
                continue
            if pcm:
                if self.first_audio_at is None:
                    self.first_audio_at = time.monotonic()
                await self._sink(pcm)


def earcon(kind: str = "heard") -> m.TtsChunk:
    """A short two-note blip the phone plays instantly, e.g. to say 'got that'."""
    notes = {"heard": (660, 880), "thinking": (523, 523)}[kind]
    t = np.linspace(0, 0.07, int(OUT_RATE * 0.07), endpoint=False)
    env = np.minimum(1, np.minimum(t / 0.01, (0.07 - t) / 0.02))  # click-free attack/release
    pcm = np.concatenate([np.sin(2 * np.pi * f * t) * env * 0.25 for f in notes])
    data = (pcm * 32767).astype(np.int16).tobytes()
    return m.TtsChunk(id=f"earcon-{kind}", seq=0, last=True, data=base64.b64encode(data).decode())


def _wav_to_pcm16(wav_bytes: bytes) -> bytes:
    """Kokoro returns 24 kHz float32 WAV (which `wave` can't read); decode with PyAV."""
    resampler = AudioResampler(format="s16", layout="mono", rate=OUT_RATE)
    out = bytearray()
    with av.open(io.BytesIO(wav_bytes)) as container:
        for frame in container.decode(audio=0):
            for r in resampler.resample(frame):
                out += r.to_ndarray().tobytes()
    for r in resampler.resample(None):  # flush
        out += r.to_ndarray().tobytes()
    return bytes(out)


def _rate(content_type: str) -> int:
    """'audio/l16;rate=24000;endianness=little-endian' -> 24000."""
    m = re.search(r"rate=(\d+)", content_type)
    return int(m.group(1)) if m else KOKORO_RATE


def _trim_silence(pcm16: bytes, rate: int) -> bytes:
    a = np.frombuffer(pcm16, dtype=np.int16)
    loud = np.flatnonzero(np.abs(a) > SILENCE)
    if not loud.size:
        return b""
    start = max(0, loud[0] - int(KEEP_BEFORE_S * rate))
    end = min(len(a), loud[-1] + int(KEEP_AFTER_S * rate))
    return a[start:end].tobytes()


def _resample(resampler: AudioResampler, pcm16: bytes | None, rate: int) -> bytes:
    """Raw PCM16 mono at `rate` -> PCM16 mono at OUT_RATE; None flushes the resampler."""
    frame = None
    if pcm16 is not None:
        frame = AudioFrame.from_ndarray(
            np.frombuffer(pcm16, dtype=np.int16).reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = rate
    return b"".join(r.to_ndarray().tobytes() for r in resampler.resample(frame))
