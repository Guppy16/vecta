"""Labelling recorded utterances: a page on the server (/label) for writing down what was
really said, so ASR models can be benchmarked on our own voices and rooms.

Recordings come from audio.Recorder (`data/sessions/<id>/utterances/`). Audio nobody has kept
is deleted after KEEP_UNLABELLED_S (see prune). Other ASR runs can
add their transcripts next to them in `utterances/hyps.jsonl` ({id, backend, text}); the
page shows them beside the live Whisper result. Labels go to `data/labels.jsonl`, one line
per save; the last line for an utterance wins.
"""

import json
import re
import time
import wave
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

PAGE = Path(__file__).parent / "static" / "label.html"
PEAK_BINS = 300
KEEP_UNLABELLED_S = 24 * 3600  # recordings are personal: only labelled ones are kept for longer
_SID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_UID = re.compile(r"^u_\d{4}$")


def router(data_dir: Path) -> APIRouter:
    r = APIRouter()
    labels_file = data_dir / "labels.jsonl"

    @r.get("/label")
    async def page() -> FileResponse:
        return FileResponse(PAGE)

    @r.get("/label/items")
    async def items() -> dict:
        out = []
        for meta_file in sorted((data_dir / "sessions").glob("*/utterances/utterances.jsonl")):
            sid, udir = meta_file.parent.parent.name, meta_file.parent
            hyps = _hyps(udir / "hyps.jsonl")
            for uid, meta in _merge(meta_file).items():
                wav = udir / f"{uid}.wav"
                if not wav.is_file():
                    continue
                duration, peaks = _peaks(wav, wav.stat().st_mtime)
                live = [{"backend": "whisper-base (live)", "text": meta.get("text"), "live": True}]
                out.append(
                    {
                        "key": f"{sid}/{uid}",
                        "session": sid,
                        "id": uid,
                        "started": meta.get("started"),
                        "kind": meta.get("kind", "speech"),
                        "duration_s": duration,
                        "peaks": peaks,
                        "hyps": live + hyps.get(uid, []),
                    }
                )
        out.sort(key=lambda u: u.get("started") or 0)
        return {"items": out, "labels": _labels(labels_file)}

    @r.get("/label/audio/{sid}/{uid}.wav")
    async def audio(sid: str, uid: str) -> FileResponse:
        if not (_SID.match(sid) and _UID.match(uid)):
            raise HTTPException(404)
        file = data_dir / "sessions" / sid / "utterances" / f"{uid}.wav"
        if not file.is_file():
            raise HTTPException(404)
        return FileResponse(file, media_type="audio/wav")

    @r.post("/label")
    async def save(body: dict) -> dict:
        label = _label(body)
        with labels_file.open("a") as f:
            f.write(json.dumps(label) + "\n")
        return label

    @r.post("/label/discard")
    async def discard(body: dict) -> dict:
        """Delete a recording's audio now (not just leave it out until the daily sweep)."""
        label = _label({**body, "exclude": True, "note": body.get("note") or "discarded"})
        label["discarded"] = True
        wav = data_dir / "sessions" / label["session"] / "utterances" / f"{label['id']}.wav"
        wav.unlink(missing_ok=True)
        with labels_file.open("a") as f:
            f.write(json.dumps(label) + "\n")
        return label

    @r.post("/label/batch")
    async def save_many(body: dict) -> list[dict]:
        """Same as /label for several utterances at once (e.g. mark all clicks as noise)."""
        labels = [_label(item) for item in body.get("items", [])]
        with labels_file.open("a") as f:
            f.writelines(json.dumps(label) + "\n" for label in labels)
        return labels

    return r


def _label(body: dict) -> dict:
    sid, uid = body.get("session", ""), body.get("id", "")
    if not (_SID.match(sid) and _UID.match(uid)):
        raise HTTPException(400, "bad session or id")
    label = {
        "session": sid,
        "id": uid,
        "reference": "" if body.get("noSpeech") else str(body.get("reference", "")).strip(),
        "noSpeech": bool(body.get("noSpeech")),
        "unsure": bool(body.get("unsure")),
        "exclude": bool(body.get("exclude")),
        "note": str(body.get("note", "")).strip(),
        "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    if not (label["reference"] or label["noSpeech"] or label["exclude"] or label["note"]):
        raise HTTPException(400, "write what was said, tick an option, or leave a comment")
    return label


def _merge(meta_file: Path) -> dict[str, dict]:
    """utterances.jsonl has one line per fact; fold them into one dict per id."""
    merged: dict[str, dict] = {}
    for line in meta_file.read_text().splitlines():
        rec = json.loads(line)
        merged.setdefault(rec["id"], {}).update(rec)
    return merged


def _hyps(file: Path) -> dict[str, list[dict]]:
    by_id: dict[str, list[dict]] = {}
    if file.is_file():
        for line in file.read_text().splitlines():
            rec = json.loads(line)
            hyp = {"backend": rec["backend"], "text": rec.get("text")}
            by_id.setdefault(rec["id"], []).append(hyp)
    return by_id


def _labels(file: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    if file.is_file():
        for line in file.read_text().splitlines():
            rec = json.loads(line)
            labels[f"{rec['session']}/{rec['id']}"] = rec
    return labels


@lru_cache(maxsize=4096)
def _peaks(wav: Path, mtime: float) -> tuple[float, list[int]]:
    """Duration and PEAK_BINS max-abs peaks (0..100, normalised per clip) for the waveform;
    `mtime` is only part of the cache key."""
    with wave.open(str(wav)) as w:
        rate, pcm = w.getframerate(), w.readframes(w.getnframes())
    a = np.abs(np.frombuffer(pcm, dtype=np.int16).astype(np.int32))
    if not a.size:
        return 0.0, []
    bins = np.array_split(a, min(PEAK_BINS, a.size))
    peaks = np.array([b.max() for b in bins], dtype=float)
    peaks = np.round(peaks / max(peaks.max(), 1) * 100).astype(int)
    return round(a.size / rate, 2), peaks.tolist()


def prune(data_dir: Path, now: float | None = None) -> list[Path]:
    """Delete recordings older than KEEP_UNLABELLED_S that no label keeps (unlabelled, or
    marked not worth keeping). Kept: a transcript, no speech, unsure, or a comment.
    Transcript text stays in utterances.jsonl and the inbox; only the audio goes."""
    now = time.time() if now is None else now
    labels = _labels(data_dir / "labels.jsonl")
    removed = []
    for wav in (data_dir / "sessions").glob("*/utterances/u_*.wav"):
        label = labels.get(f"{wav.parent.parent.name}/{wav.stem}")
        kept = (
            label
            and not label["exclude"]
            and (label["reference"] or label["noSpeech"] or label["unsure"] or label["note"])
        )
        if not kept and now - wav.stat().st_mtime > KEEP_UNLABELLED_S:
            wav.unlink(missing_ok=True)
            removed.append(wav)
    return removed
