"""Agent speech as a WebRTC audio track.

Playing TTS through the peer connection (rather than as data-channel chunks)
means the phone plays it through WebRTC's own audio path — the right output
route, and its echo canceller knows about it, so the mic doesn't pick our
voice back up. The track paces itself in real time, emitting silence when
there is nothing queued.
"""

import asyncio
import fractions
import time

import numpy as np
from aiortc.mediastreams import MediaStreamTrack
from av import AudioFrame

RATE = 16000
FRAME_MS = 20
SAMPLES = RATE * FRAME_MS // 1000  # 320 samples, 640 bytes of PCM16 per frame


class TtsTrack(MediaStreamTrack):
    kind = "audio"

    def __init__(self) -> None:
        super().__init__()
        self._queue = bytearray()
        self._lock = asyncio.Lock()
        self._start: float | None = None
        self._n = 0  # frames emitted

    def enqueue(self, pcm16: bytes) -> float:
        """Queue PCM16 mono 16 kHz; returns seconds of audio now pending."""
        self._queue += pcm16
        return len(self._queue) / (RATE * 2)

    @property
    def pending_seconds(self) -> float:
        return len(self._queue) / (RATE * 2)

    def clear(self) -> None:
        self._queue.clear()

    async def recv(self) -> AudioFrame:
        if self._start is None:
            self._start = time.monotonic()
        # keep real time: frame n is due at start + n * 20 ms
        due = self._start + self._n * FRAME_MS / 1000
        delay = due - time.monotonic()
        if delay > 0:
            await asyncio.sleep(delay)
        need = SAMPLES * 2
        chunk = bytes(self._queue[:need])
        del self._queue[:need]
        if len(chunk) < need:
            chunk += b"\x00" * (need - len(chunk))  # silence fills the gaps
        samples = np.frombuffer(chunk, dtype=np.int16).reshape(1, -1)
        frame = AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate = RATE
        frame.pts = self._n * SAMPLES
        frame.time_base = fractions.Fraction(1, RATE)
        self._n += 1
        return frame
