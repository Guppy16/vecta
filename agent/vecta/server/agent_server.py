"""
Vectaar — Agent Server (task-driven vision LLM, naive baseline)
===============================================================
The app sets a standing TASK ("tell me when you find a globe"), then sends
camera frames (snap = one, record = a stream). This server batches the
sharpest/distinct frames and asks a vision LLM whether the task condition is
met, replying with a small structured status the app can render as chat.

LLM backend: Lemonade (local Qwen3-VL on the Strix Halo), via its
OpenAI-compatible endpoint. Because it's OpenAI-shaped, switching to Gemini or
any other provider is a base_url + model + key change and nothing else.

Design choices (from our discussion):
  - STRUCTURED STATUS, not "empty response": the model returns
    {"status": "searching"|"found"|"info", "say": "..."} so "nothing to report"
    is a valid state (status=searching, say="") instead of an awkward empty msg.
  - BACKPRESSURE: one LLM call in flight at a time. Frames keep buffering during
    a call; when it returns we send the best accumulated frames. "Grouped per
    second" falls out of the model's own latency — no fixed timer to outrun.
  - DEDUP + SHARPEST: pick the sharpest recent frame that's visually distinct
    from the last one sent (dHash), so a static scene costs ~nothing and a pan
    sends fresh views.
  - DEBOUNCE: don't re-announce "found" every call while the thing stays in view.

Run ON the Lemonade machine so the LLM call is localhost:
  pip install fastapi uvicorn openai opencv-python numpy
  uvicorn vectaar_agent_server:app --host 0.0.0.0 --port 8000
Point the phone (over Tailscale) at ws://<this-box>:8000/ws
"""

import asyncio
import base64
import json
import os
import re
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from openai import OpenAI

# ---- LLM config (Lemonade / Qwen3-VL by default; swap for Gemini etc.) ----
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:13305/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "lemonade")  # unused by Lemonade
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen3-VL-4B-Instruct-GGUF")

# ---- capture/selection knobs ----
FRAMES_PER_CALL = int(os.environ.get("FRAMES_PER_CALL", "1"))  # images per LLM call
LLM_MAX_CONTEXT = int(os.environ.get("LLM_MAX_CONTEXT", "32768"))  # for the phone's ctx gauge
DEDUP_HAMMING = 8  # dHash distance below which two frames are "the same"
FOUND_DEBOUNCE = 6.0  # seconds to suppress a repeat "found" for the same task
BUFFER_MAX = 30  # recent frames kept for selection

client = OpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)

SYSTEM_TMPL = (
    "You are a real-time visual assistant for an AR app. The user has given you "
    "a standing task. You are shown one or more recent frames from their phone "
    "camera. Decide whether the task condition is met in what you can see.\n"
    "Respond with ONLY a JSON object and nothing else:\n"
    '{"status": "searching" | "found" | "info", '
    '"say": "<short text or empty>", "point": [x, y] | null}\n'
    '- "found": the task condition IS satisfied in the image(s). Put a brief, '
    'specific note in "say", and set "point" to the CENTER of the target object as '
    "normalized floats [x, y] in 0..1 (x from the left edge, y from the top edge).\n"
    '- "searching": not satisfied yet. "say" MUST be "" and "point" null.\n'
    '- "info": you must tell the user something (can\'t see clearly, need a '
    'different angle). Keep "say" short and "point" null.'
)


def dhash(gray, size=8):
    small = cv2.resize(gray, (size + 1, size))
    diff = small[:, 1:] > small[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(b)
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def sharpness(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


@dataclass
class Frame:
    ts: int
    jpeg: bytes
    sharp: float
    hash: int


@dataclass
class Session:
    task: str = ""
    buf: list = field(default_factory=list)
    in_flight: bool = False
    snap: bool = False
    last_sent_hash: int | None = None
    last_found: float = 0.0
    keyframes: list = field(default_factory=list)  # distinct frames seen this session
    history: list = field(default_factory=list)  # [{"role","content"}] text chat turns
    pending_ask: str | None = None  # a user question waiting to be answered

    def add_keyframe(self, fr):
        self.keyframes.append(fr)  # keep everything for now

    def add(self, ts, jpeg):
        arr = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_GRAYSCALE)
        if arr is None:
            return
        self.buf.append(Frame(ts, jpeg, sharpness(arr), dhash(arr)))
        self.buf = self.buf[-BUFFER_MAX:]

    def pick(self):
        """Sharpest recent frame(s) that differ from the last one we sent.
        Falls back to sharpest if nothing is 'distinct' (static scene)."""
        if not self.buf:
            return []
        cand = sorted(self.buf, key=lambda f: f.sharp, reverse=True)
        if self.last_sent_hash is not None and not self.snap:
            distinct = [f for f in cand if hamming(f.hash, self.last_sent_hash) >= DEDUP_HAMMING]
            cand = distinct or cand
        return cand[:FRAMES_PER_CALL]


def call_llm(task, frames):
    content = [{"type": "text", "text": "Latest camera frame(s). Evaluate the task now."}]
    for f in frames:
        b64 = base64.b64encode(f.jpeg).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    msgs = [
        {"role": "system", "content": SYSTEM_TMPL + "\nTask: " + task},
        {"role": "user", "content": content},
    ]
    t0 = time.time()
    try:  # some llama.cpp builds reject json_object
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            temperature=0.0,
            max_tokens=120,
            response_format={"type": "json_object"},
        )
    except Exception:
        resp = client.chat.completions.create(
            model=LLM_MODEL, messages=msgs, temperature=0.0, max_tokens=120
        )
    dt = time.time() - t0
    txt = resp.choices[0].message.content or ""
    usage = getattr(resp, "usage", None)
    return txt, dt, usage


ANSWER_SYS = (
    "You are a visual assistant answering the user's questions about what their "
    "phone camera has seen during this session. A few recent frames are attached "
    "for reference. Answer conversationally and concisely; if you can't tell from "
    "what you've seen, say so."
)


def answer_question(sess, question):
    content = [{"type": "text", "text": question}]
    for f in sess.keyframes:  # all frames seen this session
        b64 = base64.b64encode(f.jpeg).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
    msgs = (
        [{"role": "system", "content": ANSWER_SYS}]
        + sess.history[-8:]
        + [{"role": "user", "content": content}]
    )
    t0 = time.time()
    resp = client.chat.completions.create(
        model=LLM_MODEL, messages=msgs, temperature=0.2, max_tokens=300
    )
    return (resp.choices[0].message.content or ""), time.time() - t0, getattr(resp, "usage", None)


def parse_result(txt):
    """Defensive: tolerate fences / stray prose around the JSON. Returns
    (status, say, point) where point is [x, y] normalized 0..1 or None."""
    m = re.search(r"\{.*\}", txt, re.S)
    status, say, point = "info", txt.strip()[:200], None
    if m:
        try:
            o = json.loads(m.group(0))
            status = str(o.get("status", "searching")).lower()
            if status not in ("searching", "found", "info"):
                status = "info"
            say = str(o.get("say", ""))
            p = o.get("point")
            if isinstance(p, (list, tuple)) and len(p) == 2:
                x, y = float(p[0]), float(p[1])
                if x > 1.5 or y > 1.5:  # model used 0..1000 (Qwen style)
                    x, y = x / 1000.0, y / 1000.0
                if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                    point = [round(x, 4), round(y, 4)]
        except Exception:
            pass
    return status, say, point


app = FastAPI()


@app.get("/ping")
async def ping(t: int | None = None):
    # echo the client's timestamp so the phone can compute round-trip latency
    return {"t": t, "server_ms": int(time.time() * 1000)}


@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("[+] client connected")
    s = Session()
    stop = asyncio.Event()

    async def receiver():
        try:
            while True:
                msg = await websocket.receive()
                if msg.get("type") == "websocket.disconnect":
                    break
                if (b := msg.get("bytes")) is not None and len(b) >= 8:
                    s.add(int.from_bytes(b[:8], "big"), b[8:])
                elif (t := msg.get("text")) is not None:
                    try:
                        j = json.loads(t)
                    except Exception:
                        continue
                    if j.get("type") == "task":
                        s.task = j.get("task", "").strip()
                        s.buf.clear()
                        s.last_sent_hash = None
                        s.last_found = 0.0
                        # NOTE: keyframes + history are kept — session memory persists
                        # across goals so you can still ask about earlier sightings.
                        print(f"[task] {s.task!r}")
                    elif j.get("type") == "snap":
                        s.snap = True
                    elif j.get("type") == "ask":
                        s.pending_ask = j.get("q", "").strip()
                    elif j.get("type") in ("clear", "reset"):
                        s.task = ""
                        s.buf.clear()
                        s.keyframes.clear()
                        s.history.clear()
        finally:
            stop.set()

    recv = asyncio.create_task(receiver())
    try:
        while not stop.is_set():
            # answer a pending user question first (uses session keyframes + history)
            if s.pending_ask and not s.in_flight:
                q = s.pending_ask
                s.pending_ask = None
                s.in_flight = True
                try:
                    txt, dt, usage = await asyncio.to_thread(answer_question, s, q)
                    s.history.append({"role": "user", "content": q})
                    s.history.append({"role": "assistant", "content": txt})
                    s.history = s.history[-16:]
                    ct = getattr(usage, "completion_tokens", None)
                    pt = getattr(usage, "prompt_tokens", None)
                    await websocket.send_text(
                        json.dumps(
                            {
                                "type": "answer",
                                "text": txt,
                                "latency_ms": int(dt * 1000),
                                "completion_tokens": ct,
                                "tok_per_s": round(ct / dt, 1) if (ct and dt > 0) else None,
                                "ctx_tokens": pt,
                                "ctx_max": LLM_MAX_CONTEXT,
                                "n_keyframes": len(s.keyframes),
                            }
                        )
                    )
                    print(f"[ask ] {int(dt * 1000):5d}ms  {q!r}")
                except Exception as e:
                    await websocket.send_text(
                        json.dumps({"type": "answer", "text": f"(error: {e})", "latency_ms": 0})
                    )
                finally:
                    s.in_flight = False
                continue
            if not s.task or s.in_flight or not s.buf:
                await asyncio.sleep(0.03)
                continue
            frames = s.pick()
            if not frames:
                await asyncio.sleep(0.03)
                continue
            was_snap = s.snap
            s.snap = False
            s.in_flight = True
            try:
                txt, dt, usage = await asyncio.to_thread(call_llm, s.task, frames)
                status, say, point = parse_result(txt)
                s.last_sent_hash = frames[0].hash
                # debounce repeat 'found' for the same standing task
                now = time.time()
                if status == "found":
                    if now - s.last_found < FOUND_DEBOUNCE and not was_snap:
                        status = "searching"
                        say = ""
                        point = None
                    else:
                        s.last_found = now
                pt = getattr(usage, "prompt_tokens", None)
                ct = getattr(usage, "completion_tokens", None)
                out = {
                    "type": "llm",
                    "status": status,
                    "say": say,
                    "point": point,
                    "latency_ms": int(dt * 1000),
                    "prompt_tokens": pt,
                    "completion_tokens": ct,
                    "tok_per_s": round(ct / dt, 1) if (ct and dt > 0) else None,
                    "ctx_tokens": pt,
                    "ctx_max": LLM_MAX_CONTEXT,
                    "model": LLM_MODEL,
                    "ts": frames[-1].ts,
                }
                await websocket.send_text(json.dumps(out))
                print(f"[llm] {status:9s} {int(dt * 1000):5d}ms  {say!r}")
                s.add_keyframe(frames[0])  # remember for later questions
                # keep only frames newer than what we just judged
                s.buf = [f for f in s.buf if f.ts > frames[-1].ts]
            except Exception as e:
                await websocket.send_text(
                    json.dumps(
                        {
                            "type": "llm",
                            "status": "info",
                            "say": f"(LLM error: {e})",
                            "latency_ms": 0,
                        }
                    )
                )
                print(f"[llm error] {e}")
            finally:
                s.in_flight = False
    except WebSocketDisconnect:
        pass
    finally:
        recv.cancel()
        print("[-] client disconnected")
