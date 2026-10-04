"""The fast loop: a local omni model that does the listening and the talking.

Nemotron 3 Nano Omni hears each utterance as audio and decides in one short JSON
object: stay silent, look through the camera, hand over to the main agent (the
expert, who reads the same inbox and replies through `vecta say`), or answer in a
sentence. A second request with the same prompt and history transcribes the audio in
parallel; the transcript replaces the audio in the history and goes to the inbox.

The server (llama.cpp, see scripts/nemotron_server.sh) runs two slots, one for each
request, so each keeps its own cached copy of the shared prefix.
"""

import asyncio
import base64
import io
import json
import logging
import os
import re
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass, field

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

# Nemotron needs to commit to the action before it writes any words (action-first format);
# with the reply first it starts answering when it should hand over (.scratch/nemotron-hillclimb).
PERSONA = (
    "You are Vecta's voice: the fast spoken front end of an assistant on the user's phone. "
    "The user is working on something in front of the phone camera. A slower main agent, "
    "the expert, follows the same conversation; its lines also appear as assistant turns in "
    "the dialogue, and it keeps the BRIEFING up to date.\n"
    "You hear the user through the phone's microphone, which also picks up other people, "
    "background speech and noise. The latest user turn is the audio itself; earlier user "
    "turns are transcripts of what was heard.\n"
    "\n"
    "Each time the user speaks, output exactly one JSON object: "
    '{"action": "silent"}, {"action": "look"}, {"action": "expert"} or '
    '{"action": "answer", "reply": "..."}. Go down this list and take the first rule that '
    "fits.\n"
    "\n"
    "1. silent: not meant for you. Someone else's conversation, background speech, noise or "
    "no speech at all, another language out of context, a filler or hesitation sound on its "
    'own ("uh", "hmm"), a bare acknowledgement ("okay", "right", "thanks", "got it"), or a '
    "fragment that makes no sense as a request. If it is clearly said to you but too unclear "
    "to act on, answer by asking them to say it again, in a few words.\n"
    "\n"
    "2. look: about what is in view right now. You are blind: you see only when the latest "
    "message is a [tool look result] with a camera frame. Earlier descriptions in the "
    "dialogue or the briefing are stale because the camera keeps moving. So for what do you "
    "see, is something there, can you see it now, read this, what does the display or label "
    "show, where is a part: look.\n"
    "With a frame, answer only what was asked in one short sentence, and only with what you "
    "can actually read. If any part of what was asked is too small, far or blurry to read "
    "with certainty, say what you can see and ask the user to bring it closer. Never guess "
    "digits or words.\n"
    "\n"
    "3. expert: you are not the expert and never improvise instructions or explanations. "
    "This covers how to operate, set up, fix or configure something; what a button, symbol, "
    "mode, diagram or instruction means; reading or explaining instructions; any request to "
    "explain; reports that something did not work or does nothing; and replies the expert "
    "has to act on: answering a question it asked, accepting its offer, reporting what "
    "happened after a step it gave, or saying it was wrong. A filler, a thank-you or a new "
    "unrelated question after the expert spoke is not such a reply. The only steps you may "
    "say yourself are ones the BRIEFING explicitly gives you to relay.\n"
    "\n"
    '4. expert: feedback about how you or the app behave, or requests to change it ("be '
    'quicker", "don\'t say that"). You cannot change anything and must not promise to.\n'
    "\n"
    '5. answer: greetings, "can you hear me" checks, simple small talk, and simple facts or '
    'arithmetic you can answer with certainty, e.g. "what\'s 7 times 8". One short spoken '
    "sentence, no pleasantries, no offers of more help.\n"
    "\n"
    "If unsure between expert and answer, choose expert.\n"
).strip()
TRANSCRIBE = (
    "[transcribe] Write down exactly what was said in the last audio message, nothing else. "
    "If there is no speech, write [no audio]."
)
# Keeps the reply to the action-first format: only "answer" carries words.
GRAMMAR = r"""
root ::= "{\"action\": \"" (other | answer)
other ::= ("look" | "expert" | "silent") "\"}"
answer ::= "answer\", \"reply\": \"" str "\"}"
str ::= ([^"\\\x00-\x1f] | "\\" ["\\/bfnrt])*
"""
# transcripts of noise: [no audio], [BLANK_AUDIO], (music), or just punctuation
NON_SPEECH = re.compile(r"^\s*(\[[^\]]*\]|\([^)]*\)|[\s.\-]+)\s*$")
# a camera answer that asks the user to bring the thing closer: the server then keeps looking
ASKS_CLOSER = re.compile(
    r"\b(closer|clearer (view|look|picture)|can'?t (quite )?(read|see|make out)|"
    r"not (visible|readable|legible))\b",
    re.I,
)
# backstop on the talker's own words: it tends to improvise steps when asked to read a guide
INSTRUCTION_HINTS = re.compile(
    r"\b(press|hold (down|it|both)|push|tap the|turn the|switch (it|the)|"
    r"use the \w+ (button|arrow)s?|then (press|use|hold)|step (one|two|three|\d))\b",
    re.I,
)
# Trim in blocks, not a message at a time: dropping the oldest message changes the start of
# the prompt, so llama.cpp would re-read the whole history every turn instead of reusing it.
HISTORY_MAX, HISTORY_KEEP = 60, 30
DECIDE_SLOT, TRANSCRIBE_SLOT = 0, 1  # llama-server slots (-np 2)
RATE = 16000  # the listener's PCM16 mono

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
    model: str = "nemotron-omni"
    base_url: str = os.environ.get("VECTA_TALKER_URL", "http://127.0.0.1:18090/v1")
    briefing: str = "No briefing yet: the main agent has not looked at the device."
    _system_briefing: str = field(init=False)  # the one baked into the system prompt
    history: list[dict] = field(default_factory=list)
    _client: AsyncOpenAI = field(init=False)

    def __post_init__(self) -> None:
        self._system_briefing = self.briefing
        self._client = AsyncOpenAI(base_url=self.base_url, api_key="none")

    def brief(self, text: str) -> None:
        """A new briefing from the main agent. It joins the dialogue as a message rather than
        changing the system prompt, so the cached prompt prefix stays valid."""
        self.briefing = text
        self._push("user", f"[briefing update from the main agent] {text}")

    def heard(self, text: str) -> None:
        self._push("user", text)

    def hear(self, pcm16: bytes) -> asyncio.Task[str]:
        """The user said something: the audio joins the dialogue for the next decision, and
        its transcript is written in parallel. Once it is done the transcript replaces the
        audio in the history; the task returns it ("" for noise)."""
        audio = {"data": base64.b64encode(_wav(pcm16)).decode(), "format": "wav"}
        self._push("user", [{"type": "input_audio", "input_audio": audio}])
        msg = self.history[-1]
        messages = [*self._messages(), {"role": "user", "content": TRANSCRIBE}]
        return asyncio.create_task(self._transcribe(msg, messages))

    def said(self, text: str, escalate: bool = False) -> None:
        """Something said to the user on the talker's behalf (the main agent), or a hand-over
        (escalate=True) the server made for the talker."""
        self._push("assistant", _decision(text, escalate, None))

    def asked(self, tool: str) -> None:
        """Record a tool call the server made on the talker's behalf (it knew one was needed)."""
        self._push("assistant", _decision("", False, tool))

    async def turn(self, text: str | None = None, on_text: OnText | None = None) -> Reply:
        """Decide on what the user just said: `text`, or the audio already passed to hear()."""
        if text is not None:
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

    async def recheck(self, question: str, jpeg: bytes) -> str:
        """After asking the user to bring something closer: try the question again on a new
        frame. Returns the answer, or "" while it still can't tell. Nothing is added to the
        history (the caller records a positive answer with `said`)."""
        prompt = (
            f"[follow-up look] You asked the user to bring it closer. Question: {question} "
            "A new camera frame is attached. If you can now answer it with certainty, answer "
            'in one short sentence starting with "Now I can see it:"; otherwise stay silent.'
        )
        messages = [*self._messages(), {"role": "user", "content": _with_image(prompt, jpeg)}]
        # the transcribe slot: the decide slot keeps the conversation cached for the next turn
        resp = await self._client.chat.completions.create(
            **self._request(messages, 60), extra_body=_extra(TRANSCRIBE_SLOT)
        )
        reply, _, _ = parse_reply(resp.choices[0].message.content or "")
        return "" if ASKS_CLOSER.search(reply) else reply

    async def _transcribe(self, msg: dict, messages: list[dict]) -> str:
        resp = await self._client.chat.completions.create(
            **self._request(messages, 120),
            extra_body=_extra(TRANSCRIBE_SLOT, grammar=False),
        )
        text = (resp.choices[0].message.content or "").strip()
        msg["content"] = text or "[no audio]"
        return "" if NON_SPEECH.match(text) else text

    async def _complete(self, on_text: OnText | None, image: bytes | None = None) -> Reply:
        """Streams the completion; `on_text` gets the spoken reply as it is written, so speech
        can start long before the JSON is finished."""
        t0 = time.monotonic()
        messages = self._messages()
        if image is not None:
            last = messages[-1]
            messages[-1] = {"role": last["role"], "content": _with_image(last["content"], image)}
        stream = await self._client.chat.completions.create(
            **self._request(messages, 80), extra_body=_extra(DECIDE_SLOT), stream=True
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
        self._push("assistant", _decision(reply, escalate, tool))
        return Reply(reply, escalate, int((time.monotonic() - t0) * 1000), tool, first_ms)

    def _messages(self) -> list[dict]:
        return [{"role": "system", "content": self._system()}, *self.history]

    def _request(self, messages: list[dict], max_tokens: int) -> dict:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }

    def _system(self) -> str:
        return f"{PERSONA}\nBRIEFING: {self._system_briefing}"

    def _push(self, role: str, content: str | list) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > HISTORY_MAX:
            del self.history[:-HISTORY_KEEP]
            self._system_briefing = self.briefing  # the prefix changes here anyway


def _extra(slot: int, grammar: bool = True) -> dict:
    """llama-server fields: its own slot per request kind, the cached prefix reused, thinking
    off (it is slower and scored worse), and the action-first grammar for decisions."""
    body = {
        "id_slot": slot,
        "cache_prompt": True,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if grammar:
        body["grammar"] = GRAMMAR
    return body


def _with_image(text: str, jpeg: bytes) -> list[dict]:
    url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
    return [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": text}]


def _wav(pcm16: bytes) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(pcm16)
    return buf.getvalue()


def _decision(reply: str, escalate: bool, tool: str | None) -> str:
    """A turn of the talker's in its own output format, for the history."""
    if tool:
        return json.dumps({"action": tool})
    if escalate:
        return json.dumps({"action": "expert"})
    if reply:
        return json.dumps({"action": "answer", "reply": reply})
    return json.dumps({"action": "silent"})


def parse_reply(raw: str) -> tuple[str, bool, str | None]:
    """The talker's JSON as (reply, escalate, tool)."""
    m = re.search(r"\{.*\}", raw, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            action = obj.get("action")
            reply = str(obj.get("reply", "") or "").strip()
            if action == "look":
                return "", False, "look"
            return (reply if action == "answer" else ""), action == "expert", None
    return raw.strip()[:200], False, None


_REPLY_START = re.compile(r'"reply"\s*:\s*"')
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
            if not (m := _REPLY_START.search(self._raw)):
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
