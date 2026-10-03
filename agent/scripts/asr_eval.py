"""Word error rate of each ASR backend against the labels made at /label.

Labels may mark uncertain words: "(that)" is optional (no penalty either way) and
"(all|more)" accepts either word; optional words don't count towards the word total.

References come from data/labels.jsonl (the last label per utterance; ones with text,
not left out). Hypotheses: the live Whisper text in utterances.jsonl and any backend in
utterances/hyps.jsonl. Text is normalised (case, punctuation, Whisper's [BLANK_AUDIO]-
style markers, fillers like "um", contractions) so lazy casing in a label or "where's" vs
"where is" doesn't count as an error.

    uv run --no-sync python scripts/asr_eval.py [--show]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
MARKERS = re.compile(r"\[[^\]]*\]|\([^)]*\)|\*[^*]*\*|<think>.*?</think>", re.S)
FILLERS = {"um", "uh", "erm", "er", "hmm", "mm", "ah"}  # labels don't transcribe them
SPELLINGS = {  # same words, different spelling: not recognition errors
    "alright": "all right",
    "where's": "where is",
    "what's": "what is",
    "it's": "it is",
    "that's": "that is",
    "there's": "there is",
    "i'm": "i am",
    "don't": "do not",
    "doesn't": "does not",
    "didn't": "did not",
    "isn't": "is not",
    "can't": "cannot",
    "wanna": "want to",
    "gonna": "going to",
    "okay": "ok",
}


def norm(text: str | None) -> list[str]:
    text = MARKERS.sub(" ", (text or "").lower()).replace("’", "'")
    words = re.sub(r"[^a-z0-9' ]+", " ", text).split()
    words = " ".join(SPELLINGS.get(w, w) for w in words if w not in FILLERS).split()
    return words


# one reference slot: the words it accepts, and whether it may be left out
Slot = tuple[frozenset[str], bool]


def ref_slots(text: str) -> list[Slot]:
    """A label as slots: plain words, "(word)" optional, "(a|b)" either a or b."""
    slots: list[Slot] = []
    for token in re.findall(r"\([^)]*\)|[^\s()]+", text):
        if token.startswith("("):
            options = token[1:-1].split("|")
            words = frozenset(w for o in options for w in norm(o)[:1])
            optional = len(options) == 1 or any(not norm(o) for o in options)
            if words:
                slots.append((words, optional))
        else:
            slots.extend((frozenset([w]), False) for w in norm(token))
    return slots


def edits(ref: list[Slot], hyp: list[str]) -> int:
    """Word-level edit distance; an optional slot can be skipped for free."""
    prev = [0] * (len(hyp) + 1)
    for j in range(1, len(hyp) + 1):
        prev[j] = j
    for words, optional in ref:
        skip = 0 if optional else 1
        cur = [prev[0] + skip]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + skip, cur[j - 1] + 1, prev[j - 1] + (h not in words)))
        prev = cur
    return prev[-1]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(show: bool) -> None:
    labels = {}
    for rec in load_jsonl(DATA / "labels.jsonl"):
        labels[(rec["session"], rec["id"])] = rec
    refs = {k: r["reference"] for k, r in labels.items() if r["reference"] and not r["exclude"]}

    hyps: dict[str, dict] = defaultdict(dict)  # backend -> (session, id) -> text
    for meta in (DATA / "sessions").glob("*/utterances/utterances.jsonl"):
        sid = meta.parent.parent.name
        for rec in load_jsonl(meta):
            if "text" in rec:
                hyps["whisper-base (live)"][(sid, rec["id"])] = rec["text"]
        if (extra := meta.parent / "hyps.jsonl").is_file():
            for rec in load_jsonl(extra):
                hyps[rec["backend"]][(sid, rec["id"])] = rec.get("text")

    print(f"{len(refs)} labelled utterances with text\n")
    print(f"{'backend':28} {'clips':>5} {'WER':>6} {'exact':>6}")
    for backend, by_key in sorted(hyps.items()):
        keys = [k for k in refs if k in by_key]
        if not keys:
            continue
        errs = sum(edits(ref_slots(refs[k]), norm(by_key[k])) for k in keys)
        words = sum(sum(not opt for _, opt in ref_slots(refs[k])) for k in keys)
        exact = sum(edits(ref_slots(refs[k]), norm(by_key[k])) == 0 for k in keys)
        print(f"{backend:28} {len(keys):5} {errs / words:6.1%} {exact:3}/{len(keys)}")
        if show:
            for k in keys:
                if edits(ref_slots(refs[k]), norm(by_key[k])):
                    print(f"    {k[1]}  ref: {refs[k]!r}\n           hyp: {by_key[k]!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print every mismatch")
    main(ap.parse_args().show)
