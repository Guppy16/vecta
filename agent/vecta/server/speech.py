"""Agent speech: text -> Kokoro (via Lemonade) -> 16 kHz PCM16 chunks for the phone."""

from __future__ import annotations

import base64
import io
import logging
import os
import secrets

import av
import httpx
import numpy as np
from av import AudioResampler

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
