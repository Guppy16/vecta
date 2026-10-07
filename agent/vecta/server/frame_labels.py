"""Labelling camera frames: is the object a task asked about (a TV, a mug) really in view?

The frame test set is every frame a live session judged (`data/sessions/<id>/judged.jsonl`,
status "found" or "searching"). Those labels came from the live judge, so some are wrong (a
desk monitor counted as a TV); this page (/label/frames) is where they get confirmed or fixed.
Model answers to compare against come from `data/calibration/frame_preds.json`
({model: {"<session>/<file>": {"yes", "margin"}}}, written by the moondream benchmarks).
Labels go to `data/frame_labels.jsonl`, one line per save; the last line for a frame wins.
"""

import json
import re
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

PAGE = Path(__file__).parent / "static" / "frames.html"
OBJECT = re.compile(r"\b(tv|mug)\b", re.I)  # the objects the benchmark asks about
_SID = re.compile(r"^[A-Za-z0-9_-]{6,32}$")
_FILE = re.compile(r"^judged_\d{4}\.jpg$")
ANSWERS = ("yes", "no", "unsure")


def router(data_dir: Path) -> APIRouter:
    r = APIRouter()
    labels_file = data_dir / "frame_labels.jsonl"

    @r.get("/label/frames")
    async def page() -> FileResponse:
        return FileResponse(PAGE)

    @r.get("/label/frames/items")
    async def items() -> dict:
        preds = _read_json(data_dir / "calibration" / "frame_preds.json")
        out = []
        for judged in sorted((data_dir / "sessions").glob("*/judged.jsonl")):
            sid = judged.parent.name
            for line in judged.read_text().splitlines():
                frame = json.loads(line)
                if not (m := OBJECT.search(frame["task"])):
                    continue
                key = f"{sid}/{frame['file']}"
                out.append(
                    {
                        "key": key,
                        "session": sid,
                        "file": frame["file"],
                        "task": frame["task"],
                        "object": m.group(1).lower(),
                        "judged": frame["status"] == "found",
                        "answers": {name: p[key] for name, p in preds.items() if key in p},
                    }
                )
        return {"items": out, "models": list(preds), "labels": _labels(labels_file)}

    @r.get("/label/frames/img/{sid}/{file}")
    async def image(sid: str, file: str) -> FileResponse:
        if not (_SID.match(sid) and _FILE.match(file)):
            raise HTTPException(404)
        path = data_dir / "sessions" / sid / file
        if not path.is_file():
            raise HTTPException(404)
        return FileResponse(path, media_type="image/jpeg")

    @r.post("/label/frames")
    async def save(body: dict) -> dict:
        key = str(body.get("key", ""))
        sid, _, file = key.partition("/")
        if not (_SID.match(sid) and _FILE.match(file)):
            raise HTTPException(422, "key must be <session>/judged_NNNN.jpg")
        if body.get("present") not in ANSWERS:
            raise HTTPException(422, f"present must be one of {ANSWERS}")
        label = {
            "key": key,
            "present": body["present"],
            "note": str(body.get("note") or "").strip(),
            "updatedAt": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with labels_file.open("a") as f:
            f.write(json.dumps(label) + "\n")
        return label

    return r


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.is_file() else {}


def _labels(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text().splitlines():
        if line.strip():
            label = json.loads(line)
            out[label["key"]] = label
    return out
