"""The fast loop: a small local LLM that does the talking.

Every transcribed utterance gets one call (~0.4 s with the system prompt cached).
The talker only knows the dialogue and a BRIEFING the main agent keeps up to date;
it answers in a sentence or two, stays silent for background chatter, and when a
question needs the camera or real reasoning it says so and flags an escalation
for the main agent, who reads the same inbox and replies through `vecta say`.
"""

import base64
import json
import logging
import os
import re
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

PERSONA = (
    "You are Vecta's voice: a terse spoken assistant on the user's phone, helping with a "
    "task in front of the camera. You see only when a camera frame is attached to a message; "
    "otherwise you know only the BRIEFING, the dialogue and earlier answers — never describe or "
    "invent device details, readings or labels you have not seen. "
    "Reply in one or two short spoken sentences. Always answer the user when they speak to you, "
    "including greetings, 'can you hear me' checks and questions about what you can do. Reply "
    "with an empty string only when the words are clearly not meant for you (someone else's "
    "conversation, noise, a cut-off half sentence). "
    'Tool: to see what the camera shows right now, answer exactly {"tool": "look"}; the frame '
    "arrives in a message starting with [tool look result]. Any question about what is visible, "
    "what something looks like, or what is on a display MUST start with that tool call unless a "
    "frame is already attached. With a frame, answer what was asked in one short sentence "
    "about the thing asked about; don't describe the whole scene. "
    "You are NOT the expert: you never explain how to operate, set, program or fix a device, "
    "what a button or symbol does, or why something happens — "
    "and you never offer to. For any such question reply with a short holding line "
    '(e.g. "Let me work that out properly, one moment.") and set escalate=true; the main agent '
    "answers. The briefing may contain steps the main agent wants relayed: those you may say. "
    'Otherwise answer ONLY as JSON: {"reply": "...", "escalate": false}.'
)
# server-side backstop: how-to questions are the main agent's, even if the talker answers
HOWTO_HINTS = re.compile(
    r"\b(how (do|can|should|to)|what does|what is th(is|at) (button|symbol|icon|light)|set|"
    r"program|schedule|adjust|change|turn (it )?(on|off|up|down)|why (does|is|won.t|doesn.t)|"
    r"explain|instructions?|steps?)\b",
    re.I,
)
HOLDING_LINE = "Let me work that out properly, one moment."
# server-side backstop: these utterances get the look tool even if the model forgets to ask
LOOK_HINTS = re.compile(
    r"\b(see|seeing|look|looking|show|showing|display|screen|camera|in view|what is this|"
    r"what's this|read|reading|says|label)\b",
    re.I,
)
HISTORY_TURNS = 24

OnText = Callable[[str], None]


@dataclass(frozen=True)
class Reply:
    text: str
    escalate: bool
    latency_ms: int
    tool: str | None = None  # the talker wants a tool run first (e.g. "look")
    first_text_ms: int | None = None  # when the first words of the reply were available


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

    def asked(self, tool: str) -> None:
        """Record a tool call the server made on the talker's behalf (it knew one was needed)."""
        self._push("assistant", json.dumps({"tool": tool}))

    async def turn(self, text: str, on_text: OnText | None = None) -> Reply:
        self.heard(text)
        return await self._complete(on_text)

    async def tool_result(self, tool: str, result: str, on_text: OnText | None = None) -> Reply:
        """Feed a tool's output back and let the talker answer with it."""
        self._push("user", f"[tool {tool} result] {result}")
        return await self._complete(on_text)

    async def look_result(self, jpeg: bytes, on_text: OnText | None = None) -> Reply:
        """Answer the pending question from a camera frame. The frame goes to the model with
        this one request only; the history keeps a text placeholder, so later turns stay small
        and the cached prefix stays valid."""
        self._push("user", "[tool look result] camera frame attached")
        return await self._complete(on_text, image=jpeg)

    async def _complete(self, on_text: OnText | None, image: bytes | None = None) -> Reply:
        """Streams the completion; `on_text` gets the spoken reply as it is written, so speech
        can start long before the JSON is finished."""
        t0 = time.monotonic()
        messages = [{"role": "system", "content": self._system()}, *self.history]
        if image is not None:
            url = "data:image/jpeg;base64," + base64.b64encode(image).decode()
            last = messages[-1]
            messages[-1] = {
                "role": last["role"],
                "content": [
                    {"type": "image_url", "image_url": {"url": url}},
                    {"type": "text", "text": last["content"]},
                ],
            }
        stream = await self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
            max_tokens=80,
            response_format={"type": "json_object"},
            extra_body={
                "cache_prompt": True,  # the prefix is append-only; llama.cpp reuses it
                # Qwen3.6 thinks by default, which spends the whole token budget before the JSON
                "chat_template_kwargs": {"enable_thinking": False},
            },
            stream=True,
        )
        raw, first_ms, timings = "", None, {}
        extractor = ReplyExtractor()
        async for chunk in stream:
            timings = (chunk.model_extra or {}).get("timings") or timings  # llama.cpp, last chunk
            delta = chunk.choices[0].delta.content if chunk.choices else None
            if not delta:
                continue
            raw += delta
            if (text := extractor.feed(delta)) and on_text:
                first_ms = first_ms or int((time.monotonic() - t0) * 1000)
                on_text(text)
        log.info(
            "talker llm: prompt %s tok (%s cached) in %s ms, %s tok out",
            timings.get("prompt_n"),
            timings.get("cache_n"),
            round(timings.get("prompt_ms", 0)),
            timings.get("predicted_n"),
        )
        reply, escalate, tool = parse_reply(raw)
        if extractor.started:
            reply = extractor.text.strip()  # what was actually spoken, even if the JSON broke
        elif reply and not tool and on_text:
            first_ms = int((time.monotonic() - t0) * 1000)
            on_text(reply)  # keys came in another order; speak it whole
        if tool:
            self._push("assistant", json.dumps({"tool": tool}))
        else:
            self._push("assistant", json.dumps({"reply": reply, "escalate": escalate}))
        return Reply(reply, escalate, int((time.monotonic() - t0) * 1000), tool, first_ms)

    def _system(self) -> str:
        return f"{PERSONA}\nBRIEFING: {self.briefing}"

    def _push(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        del self.history[:-HISTORY_TURNS]


def parse_reply(raw: str) -> tuple[str, bool, str | None]:
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            tool = obj.get("tool")
            return (
                str(obj.get("reply", "") or "").strip(),
                bool(obj.get("escalate", False)),
                str(tool) if tool else None,
            )
        except json.JSONDecodeError:
            pass
    return raw.strip()[:200], False, None


_REPLY_START = re.compile(r'^\s*\{\s*"reply"\s*:\s*"')
_ESCAPES = {"n": "\n", "t": "\t", "r": "", "b": "", "f": ""}


class ReplyExtractor:
    """Pulls the "reply" string out of the talker's JSON while it is still streaming.

    feed() returns the newly decoded reply text (JSON escapes resolved); a half-received
    escape sequence waits for the next delta.
    """

    def __init__(self) -> None:
        self._raw = ""
        self._pos: int | None = None  # index into _raw of the next undecoded reply char
        self.text = ""
        self.done = False

    @property
    def started(self) -> bool:
        return self._pos is not None

    def feed(self, delta: str) -> str:
        self._raw += delta
        if self.done:
            return ""
        if self._pos is None:
            if not (m := _REPLY_START.match(self._raw)):
                return ""
            self._pos = m.end()
        raw, i, out = self._raw, self._pos, []
        while i < len(raw):
            c = raw[i]
            if c == '"':
                self.done = True
                break
            if c != "\\":
                out.append(c)
                i += 1
                continue
            if i + 1 >= len(raw):
                break
            e = raw[i + 1]
            if e == "u":
                if i + 6 > len(raw):
                    break
                try:
                    out.append(chr(int(raw[i + 2 : i + 6], 16)))
                except ValueError:
                    pass
                i += 6
                continue
            out.append(_ESCAPES.get(e, e))
            i += 2
        self._pos = i
        text = "".join(out)
        self.text += text
        return text
