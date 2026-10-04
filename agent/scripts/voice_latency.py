"""Voice-to-voice latency of the fast loop, measured on the box without a phone.

For each question: Kokoro speaks it (standing in for the user), then
talker (hearing the audio) -> Kokoro, timing until the reply is first audible (silence at
the start of the audio counts as waiting), both the old way (whole reply, then
whole synthesis, untrimmed) and streaming (SpeechStream).
The listener's end-of-speech wait is added on top: END_SILENCE_MS for the old
way, and for streaming the speculative start (SPECULATE_SILENCE_MS), gated by
END_SILENCE_MS. Network and the phone's jitter buffer are not included.

    uv run --no-sync python scripts/voice_latency.py
"""

import asyncio
import statistics
import time

import numpy as np

from vecta.server.audio import END_SILENCE_MS, SPECULATE_SILENCE_MS
from vecta.server.speech import Speaker, SpeechStream
from vecta.server.talker import Talker

QUESTIONS = [
    "Hey, can you hear me?",
    "Okay, I'm in the hallway now, I'll walk over to the thermostat.",
    "What's the best way to describe what a thermostat does, in a few words?",
    "Thanks, that's really helpful. Let's carry on in a minute.",
]
BRIEFING = "The user is setting up a Danfoss TP5000 programmable room thermostat."


async def main() -> None:
    tts = Speaker()
    speech = {q: await tts.synthesize(q) for q in QUESTIONS}
    rows: dict[str, list[int]] = {"old": [], "stream": []}
    for mode in ("old", "stream") * 3:
        talker = Talker(briefing=BRIEFING)
        for q in QUESTIONS:
            t0 = time.monotonic()
            heard = talker.hear(speech[q])
            if mode == "old":
                reply = await talker.turn()
                lead = 0.0
                if reply.text:
                    lead = _lead_silence(await tts.synthesize(reply.text))
                first = time.monotonic() + lead
            else:
                leads: list[float] = []

                async def sink(pcm: bytes, leads: list[float] = leads) -> None:
                    leads.append(_lead_silence(pcm))

                voice = SpeechStream(tts, sink)
                reply = await talker.turn(on_text=voice.write)
                await voice.finish()
                first = (voice.first_audio_at or time.monotonic()) + (leads[0] if leads else 0)
            text = await heard
            if not reply.text:
                continue
            work = int((first - t0) * 1000)
            if mode == "old":
                ms = END_SILENCE_MS + work
            else:  # replying started at the short pause, heard once the pause is long enough
                ms = max(END_SILENCE_MS, SPECULATE_SILENCE_MS + work)
            rows[mode].append(ms)
            print(
                f"{mode:6} v2v {ms:5} ms  first text {reply.first_text_ms} ms  "
                f"heard {text!r} -> {reply.text!r}"
            )
    for mode, xs in rows.items():
        print(f"{mode}: median voice-to-voice {statistics.median(xs)} ms over {len(xs)} replies")


def _lead_silence(pcm16: bytes, rate: int = 16000) -> float:
    loud = np.flatnonzero(np.abs(np.frombuffer(pcm16, dtype=np.int16)) > 500)
    return loud[0] / rate if loud.size else 0.0


if __name__ == "__main__":
    asyncio.run(main())
