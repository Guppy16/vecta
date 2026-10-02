"""Labelling recorded utterances: a small page on the server for writing down what was
really said, so ASR models can be benchmarked on our own voices and rooms.

Recordings come from audio.Recorder (`data/sessions/<id>/utterances/`); labels go to
`data/labels.jsonl`, one line per save (the last one for an utterance wins).
"""

import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

PAGE = Path(__file__).parent / "static" / "label.html"
KINDS = {"speech", "not_speech", "not_for_agent", "unclear"}
_ID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_UTT = re.compile(r"^u_\d{4}$")


def router(data_dir: Path) -> APIRouter:
    r = APIRouter()
    labels_file = data_dir / "labels.jsonl"

    @r.get("/label")
    async def page() -> FileResponse:
        return FileResponse(PAGE)

    @r.get("/label/items")
    async def items() -> list[dict]:
        labels = _read_labels(labels_file)
        out = []
        for meta_file in (data_dir / "sessions").glob("*/utterances/utterances.jsonl"):
            sid = meta_file.parent.parent.name
            for uid, meta in _merge(meta_file).items():
                out.append({"session": sid, **meta, "label": labels.get((sid, uid))})
        return sorted(out, key=lambda u: u.get("started", 0), reverse=True)

    @r.get("/label/audio/{sid}/{uid}.wav")
    async def audio(sid: str, uid: str) -> FileResponse:
        if not (_ID.match(sid) and _UTT.match(uid)):
            raise HTTPException(404)
        file = data_dir / "sessions" / sid / "utterances" / f"{uid}.wav"
        if not file.is_file():
            raise HTTPException(404)
        return FileResponse(file, media_type="audio/wav")

    @r.post("/label")
    async def save(body: dict) -> dict:
        sid, uid, kind = body.get("session", ""), body.get("id", ""), body.get("kind", "")
        if not (_ID.match(sid) and _UTT.match(uid)) or kind not in KINDS:
            raise HTTPException(400, "bad session, id or kind")
        line = {
            "session": sid,
            "id": uid,
            "kind": kind,
            "transcript": str(body.get("transcript", "")).strip(),
            "ts": round(time.time(), 3),
        }
        with labels_file.open("a") as f:
            f.write(json.dumps(line) + "\n")
        return line

    return r


def _merge(meta_file: Path) -> dict[str, dict]:
    """utterances.jsonl has one line per fact; fold them into one dict per id."""
    merged: dict[str, dict] = {}
    for line in meta_file.read_text().splitlines():
        rec = json.loads(line)
        merged.setdefault(rec["id"], {}).update(rec)
    return merged


def _read_labels(file: Path) -> dict[tuple[str, str], dict]:
    if not file.is_file():
        return {}
    labels = {}
    for line in file.read_text().splitlines():
        rec = json.loads(line)
        labels[(rec["session"], rec["id"])] = rec
    return labels
