"""Local vision-language model via Lemonade's OpenAI-compatible endpoint.

Two calls: `judge` (is the standing task satisfied in this frame?) returning a
small structured verdict, and `answer` (free-form question over recent frames).
Both are cheap enough to run continuously on the box's GPU.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import time
from dataclasses import dataclass

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Short on purpose: every token here is prefilled on every call (~1 ms each on the box).
JUDGE_SYSTEM = (
    "Real-time camera watch. Standing task below. Look at the frame and answer with ONLY "
    'JSON: {"status": "searching"|"found"|"info", "say": "", "count": 0}. '
    "found = task condition is visible: say what/where briefly, count = matching things. "
    'searching = not yet: say "" and count 0. info = you must tell the user something '
    "(unclear, need another angle): say it briefly."
)

ANSWER_SYSTEM = (
    "You are a visual assistant answering the user's questions about what their phone "
    "camera has seen during this session. Recent frames are attached, oldest first. "
    "Answer conversationally and concisely; if you can't tell from what you've seen, say so."
)


@dataclass(frozen=True)
class Verdict:
    status: str  # searching | found | info
    say: str
    latency_ms: int
    count: int = 0  # matching instances in view, when found


class Vlm:
    def __init__(
        self,
        base_url: str = os.environ.get("VECTA_LLM_BASE_URL", "http://127.0.0.1:13305/api/v1"),
        model: str = os.environ.get("VECTA_LLM_MODEL", "Qwen3-VL-4B-Instruct-GGUF"),
    ) -> None:
        self.model = model
        self._client = AsyncOpenAI(base_url=base_url, api_key="lemonade")

    async def judge(self, task: str, jpeg: bytes) -> Verdict:
        t0 = time.monotonic()
        text = await self.chat(
            system=f"{JUDGE_SYSTEM}\nTask: {task}",
            user_text="Latest camera frame. Evaluate the task now.",
            jpegs=[jpeg],
            max_tokens=120,
            json_mode=True,
        )
        status, say, count = parse_verdict(text)
        return Verdict(status, say, int((time.monotonic() - t0) * 1000), count)

    async def answer(self, question: str, jpegs: list[bytes], task: str | None) -> str:
        system = ANSWER_SYSTEM + (f"\nThe user's standing task is: {task}" if task else "")
        return (await self.chat(system, question, jpegs, max_tokens=300)).strip()

    async def chat(
        self,
        system: str,
        user_text: str,
        jpegs: list[bytes],
        max_tokens: int,
        json_mode: bool = False,
    ) -> str:
        content: list[dict] = [{"type": "text", "text": user_text}]
        for j in jpegs:
            b64 = base64.b64encode(j).decode()
            content.append(
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": content}]
        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        try:
            resp = await self._client.chat.completions.create(**kwargs)
        except Exception:
            if not json_mode:
                raise
            kwargs.pop("response_format")  # some llama.cpp builds reject json_object
            resp = await self._client.chat.completions.create(**kwargs)
        return resp.choices[0].message.content or ""


def parse_verdict(text: str) -> tuple[str, str, int]:
    """Tolerate code fences and stray prose around the JSON. Unparseable -> info."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            status = str(obj.get("status", "info"))
            if status in ("searching", "found", "info"):
                count = obj.get("count", 1 if status == "found" else 0)
                count = (
                    int(count)
                    if isinstance(count, int | float)
                    else (1 if status == "found" else 0)
                )
                return (
                    status,
                    str(obj.get("say", "") or ""),
                    max(count, 1 if status == "found" else 0),
                )
        except json.JSONDecodeError, ValueError:
            pass
    return "info", text.strip()[:200], 0
