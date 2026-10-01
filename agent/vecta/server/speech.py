"""Agent speech: text -> Kokoro (via Lemonade) -> 16 kHz PCM16 chunks for the phone."""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets
import wave

import httpx
import numpy as np

from vecta.protocol import messages as m

log = logging.getLogger(__name__)

OUT_RATE = 16000
CHUNK_BYTES = 24000  # 0.75 s of PCM16 at 16 kHz; ~32 KB as base64, well under channel limits


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

    async def chunks(self, text: str) -> list[m.TtsChunk]:
        pcm = await self.synthesize(text)
        uid = secrets.token_urlsafe(6)
        parts = [pcm[i : i + CHUNK_BYTES] for i in range(0, len(pcm), CHUNK_BYTES)] or [b""]
        return [
            m.TtsChunk(id=uid, seq=i, last=i == len(parts) - 1, data=base64.b64encode(p).decode())
            for i, p in enumerate(parts)
        ]


def _wav_to_pcm16(wav_bytes: bytes) -> bytes:
    """Kokoro returns 24 kHz float32 WAV; the phone wants 16 kHz int16."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        rate, width, channels = w.getframerate(), w.getsampwidth(), w.getnchannels()
        raw = w.readframes(w.getnframes())
    if width == 4:  # IEEE float
        samples = np.frombuffer(raw, dtype=np.float32)
    else:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    if rate != OUT_RATE:  # linear resample is fine for speech
        n = int(len(samples) * OUT_RATE / rate)
        samples = np.interp(np.linspace(0, len(samples) - 1, n), np.arange(len(samples)), samples)
    return (np.clip(samples, -1, 1) * 32767).astype(np.int16).tobytes()
