"""Always-on listening: mic track -> VAD segments -> Whisper -> inbox.

The phone streams its microphone as a WebRTC audio track. We resample to
16 kHz mono, cut it into utterances with a simple VAD, transcribe each one
with Lemonade's Whisper endpoint, and append the text to the session inbox.
Deciding whether an utterance deserves a response is the agent's job, not
ours — noise and half-sentences get logged too.
"""

from __future__ import annotations

import io
import logging
import os
import re
import time
import wave
from collections.abc import Callable

import httpx
import webrtcvad
from aiortc.mediastreams import MediaStreamError, MediaStreamTrack
from av import AudioResampler

log = logging.getLogger(__name__)

RATE = 16000
FRAME_MS = 30  # webrtcvad accepts 10/20/30 ms frames
FRAME_BYTES = RATE * FRAME_MS // 1000 * 2  # PCM16 mono
MIN_UTTERANCE_MS = 400
END_SILENCE_MS = 350  # end of utterance; shorter = snappier, risks splitting slow sentences
MAX_UTTERANCE_MS = 15000
# Whisper's non-speech tokens: [BLANK_AUDIO], [typing], (music), or just punctuation.
NON_SPEECH = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\)|[\s.\-]+)\s*$")


class Transcriber:
    def __init__(
        self,
        base_url: str = os.environ.get("VECTA_LLM_BASE_URL", "http://127.0.0.1:13305/api/v1"),
        model: str = os.environ.get("VECTA_STT_MODEL", "Whisper-Base"),
    ) -> None:
        self._client = httpx.AsyncClient(base_url=base_url, timeout=60)
        self.model = model

    async def transcribe(self, pcm16: bytes) -> str:
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(RATE)
            w.writeframes(pcm16)
        r = await self._client.post(
            "/audio/transcriptions",
            data={"model": self.model},
            files={"file": ("u.wav", buf.getvalue(), "audio/wav")},
        )
        r.raise_for_status()
        return r.json().get("text", "").strip()


class Listener:
    """Consumes one audio track; calls `on_text(text, started_at)` per utterance.

    `muted_until` (epoch seconds) is set by the server while the agent is speaking,
    so the phone's own speaker output isn't transcribed back as the user.
    """

    def __init__(
        self,
        transcriber: Transcriber,
        on_text: Callable[[str, float], None],
        aggressiveness: int = 2,
    ) -> None:
        self._stt = transcriber
        self._on_text = on_text
        self.muted_until = 0.0
        self._started_at = 0.0
        self._vad = webrtcvad.Vad(aggressiveness)
        self._resampler = AudioResampler(format="s16", layout="mono", rate=RATE)
        self._pending = b""  # resampled bytes not yet cut into VAD frames
        self._utterance = bytearray()
        self._voiced_ms = 0
        self._silence_ms = 0

    async def mute_for(self, seconds: float) -> None:
        """The agent is about to speak: finish what the user was saying, ignore the rest."""
        self.muted_until = time.time() + seconds
        if self._utterance:
            await self._flush()

    async def run(self, track: MediaStreamTrack) -> None:
        try:
            while True:
                for frame in self._resampler.resample(await track.recv()):
                    self._pending += frame.to_ndarray().tobytes()
                    await self._drain()
        except MediaStreamError:
            log.info("audio track ended")

    async def _drain(self) -> None:
        while len(self._pending) >= FRAME_BYTES:
            chunk, self._pending = self._pending[:FRAME_BYTES], self._pending[FRAME_BYTES:]
            if self._vad.is_speech(chunk, RATE):
                if not self._utterance:
                    self._started_at = time.time()
                self._utterance += chunk
                self._voiced_ms += FRAME_MS
                self._silence_ms = 0
            elif self._utterance:
                self._utterance += chunk  # keep trailing silence so words aren't clipped
                self._silence_ms += FRAME_MS
            ms = len(self._utterance) * 1000 // (RATE * 2)
            if self._utterance and (self._silence_ms >= END_SILENCE_MS or ms >= MAX_UTTERANCE_MS):
                await self._flush()

    async def _flush(self) -> None:
        pcm, voiced, started = bytes(self._utterance), self._voiced_ms, self._started_at
        self._utterance, self._voiced_ms, self._silence_ms = bytearray(), 0, 0
        if voiced < MIN_UTTERANCE_MS:
            return  # a click, a cough
        if started > self.muted_until - 0.5 and started < self.muted_until:
            return  # that was us talking
        try:
            text = await self._stt.transcribe(pcm)
        except Exception as e:
            log.warning("stt failed: %s", e)
            return
        if text and not NON_SPEECH.match(text):
            self._on_text(text, started)
