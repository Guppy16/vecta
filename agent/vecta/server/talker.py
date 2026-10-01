"""The fast loop: a small local LLM that does the talking.

Every transcribed utterance gets one call (~0.4 s with the system prompt cached).
The talker only knows the dialogue and a BRIEFING the main agent keeps up to date;
it answers in a sentence or two, stays silent for background chatter, and when a
question needs the camera or real reasoning it says so and flags an escalation
for the main agent, who reads the same inbox and replies through `vecta say`.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

PERSONA = (
    "You are Vecta's voice: a terse spoken assistant on the user's phone, helping with a "
    "task in front of the camera. You know only the BRIEFING and the dialogue. Reply in one or "
    "two short spoken sentences. Reply with an empty string when the user is not talking to you "
    "(background conversation, noise, half sentences). If answering needs looking at the camera, "
    "or reasoning you can't do from the briefing, say you'll take a look and set escalate=true "
    "— the main agent will follow up. Never invent device details. "
    'Answer ONLY as JSON: {"reply": "...", "escalate": false}.'
)
HISTORY_TURNS = 24


@dataclass(frozen=True)
class Reply:
    text: str
    escalate: bool
    latency_ms: int


@dataclass
class Talker:
    model: str = os.environ.get("VECTA_TALKER_MODEL", "Qwen3.6-35B-A3B-MTP-GGUF")
    base_url: str = os.environ.get("VECTA_LLM_BASE_URL", "http://127.0.0.1:13305/api/v1")
    briefing: str = "No briefing yet: the main agent has not looked at the device."
    history: list[dict[str, str]] = field(default_factory=list)
    _client: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="lemonade")

    def heard(self, text: str) -> None:
        self._push("user", text)

    def said(self, text: str) -> None:
        """Something the main agent said to the user; the talker shouldn't repeat it."""
        self._push("assistant", json.dumps({"reply": text, "escalate": False}))

    async def turn(self, text: str) -> Reply:
        self.heard(text)
        t0 = time.monotonic()
        kwargs: dict = {
            "model": self.model,
            "messages": [{"role": "system", "content": self._system()}, *self.history],
            "temperature": 0.2,
            "max_tokens": 80,
            "response_format": {"type": "json_object"},
            "extra_body": {"cache_prompt": True},  # the prefix is append-only; llama.cpp reuses it
        }
        resp = await self._client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        reply, escalate = parse_reply(raw)
        self._push("assistant", json.dumps({"reply": reply, "escalate": escalate}))
        return Reply(reply, escalate, int((time.monotonic() - t0) * 1000))

    def _system(self) -> str:
        return f"{PERSONA}\nBRIEFING: {self.briefing}"

    def _push(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        del self.history[:-HISTORY_TURNS]


def parse_reply(raw: str) -> tuple[str, bool]:
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            return str(obj.get("reply", "") or "").strip(), bool(obj.get("escalate", False))
        except json.JSONDecodeError:
            pass
    return raw.strip()[:200], False
