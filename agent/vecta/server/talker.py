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

# Decision order tuned on data/calibration (scripts/talker_eval.py): 49-50/54 raw.
PERSONA = (
    "You are Vecta's voice: the fast spoken front end of an assistant on the user's phone. "
    "The user is working on something in front of the phone camera. A slower main agent, "
    "the expert, follows the same conversation; its lines also appear as assistant turns in "
    "the dialogue, and it keeps the BRIEFING up to date.\n"
    "You hear the user through the phone's microphone: their words reach you as a speech "
    'transcript, which is often garbled (similar-sounding words, e.g. "cable" heard as '
    '"table"). Read each line for what they most likely said.\n'
    "\n"
    "Each time the user speaks, decide in this order and output exactly one JSON object:\n"
    "\n"
    "1. Not meant for you? Someone else's conversation, background speech, transcription "
    'noise such as "[no audio]" or bracketed sounds, another language out of context, a '
    'filler or hesitation sound on its own ("uh", "hmm"), a bare acknowledgement ("okay", '
    '"right", "thanks", "got it"), or a fragment that makes no sense as a request: '
    '{"reply": "", "escalate": false}\n'
    "If it is clearly said to you but too garbled to act on, ask them to say it again in a "
    "few words.\n"
    "\n"
    "2. About what is in view right now? You are blind: you see only when the latest "
    "message is a [tool look result] with a camera frame. Earlier descriptions in the "
    "dialogue or the briefing are stale because the camera keeps moving. So for what do you "
    "see, is something there, can you see it now, read this, what does the display or label "
    'show, where is a part: {"tool": "look"}\n'
    "With a frame, answer only what was asked in one short sentence, and only with what you "
    "can actually read. If any part of what was asked is too small, far or blurry to read "
    "with certainty, say what you can see and ask the user to bring it closer. Never guess "
    "digits or words.\n"
    "\n"
    "3. Needs the expert? You are not the expert and never improvise instructions or "
    "explanations. This covers how to operate, set up, fix or configure something; what a "
    "button, symbol, mode, diagram or instruction means; reading or explaining "
    "instructions; any request to explain; reports that something did not work or does "
    "nothing; and replies the expert has to act on: answering a question it asked, "
    "accepting its offer, reporting what happened after a step it gave, or saying it was "
    "wrong. A filler, a thank-you or a new unrelated question after the expert spoke is not "
    "such a reply. Output exactly:\n"
    '{"reply": "Let me work that out properly, one moment.", "escalate": true}\n'
    "escalate=true is what calls the expert; the holding line alone does nothing. Always "
    "send them together, even if earlier turns show otherwise. The only steps you may say "
    "yourself are ones the BRIEFING explicitly gives you to relay.\n"
    "\n"
    '4. Feedback about how you or the app behave, or requests to change it ("be quicker", '
    '"don\'t say that"): you cannot change anything and must not promise to; pass it on '
    'silently: {"reply": "", "escalate": true}\n'
    "\n"
    '5. Otherwise (greetings, "can you hear me" checks, simple small talk, and simple facts '
    'or arithmetic you can answer with certainty, e.g. "what\'s 7 times 8"): {"reply": "<one '
    'short spoken sentence, no pleasantries, no offers of more help>", "escalate": false}\n'
    "\n"
    "If unsure between 3 and 5, choose 3.\n"
).strip()
# server-side backstop: how-to questions are the main agent's, even if the talker answers
HOWTO_HINTS = re.compile(
    r"\b(how (do|can|should|to)|what does|what is th(is|at) (button|symbol|icon|light)|set|"
    r"program|schedule|adjust|change|turn (it )?(on|off|up|down)|why (does|is|won.t|doesn.t)|"
    r"explain|instructions?|steps?)\b",
    re.I,
)
HOLDING_LINE = "Let me work that out properly, one moment."
# backstop on the talker's own words: it tends to improvise steps when asked to read a guide
INSTRUCTION_HINTS = re.compile(
    r"\b(press|hold (down|it|both)|push|tap the|turn the|switch (it|the)|"
    r"use the \w+ (button|arrow)s?|then (press|use|hold)|step (one|two|three|\d))\b",
    re.I,
)
# server-side backstop: these utterances get the look tool even if the model forgets to ask
LOOK_HINTS = re.compile(
    r"\b(see|seeing|look|looking|show|showing|display|screen|camera|in view|what is this|"
    r"what's this|read|reading|says|label)\b",
    re.I,
)
# Trim in blocks, not a message at a time: dropping the oldest message changes the start of
# the prompt, so llama.cpp would re-read the whole history every turn instead of reusing it.
HISTORY_MAX, HISTORY_KEEP = 60, 30

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
    _system_briefing: str = field(init=False)  # the one baked into the system prompt
    history: list[dict[str, str]] = field(default_factory=list)
    _client: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._system_briefing = self.briefing
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="lemonade")

    def brief(self, text: str) -> None:
        """A new briefing from the main agent. It joins the dialogue as a message rather than
        changing the system prompt, so the cached prompt prefix stays valid."""
        self.briefing = text
        self._push("user", f"[briefing update from the main agent] {text}")

    def heard(self, text: str) -> None:
        self._push("user", text)

    def said(self, text: str, escalate: bool = False) -> None:
        """Something said to the user on the talker's behalf (the main agent, or the server's
        holding line, which must be logged as escalate=true or the model learns to say the
        holding line without escalating)."""
        self._push("assistant", json.dumps({"reply": text, "escalate": escalate}))

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
        return f"{PERSONA}\nBRIEFING: {self._system_briefing}"

    def _push(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > HISTORY_MAX:
            del self.history[:-HISTORY_KEEP]
            self._system_briefing = self.briefing  # the prefix changes here anyway


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
