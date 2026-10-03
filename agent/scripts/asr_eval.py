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
# chat-style models sometimes answer silence with a refusal; that means "no speech"
REFUSAL = re.compile(
    r"^\s*i'?m sorry,? but i (can'?t|cannot|can not) provide (a|the) transcription", re.I
)
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
    text = (text or "").replace("’", "'")
    if REFUSAL.match(text):
        return []
    text = MARKERS.sub(" ", text.lower())
    words = re.sub(r"[^a-z0-9' ]+", " ", text).split()
    words = " ".join(SPELLINGS.get(w, w) for w in words if w not in FILLERS).split()
    return words


# One reference slot: the word sequences it accepts. A plain word accepts itself; "(that)"
# accepts "that" or nothing; "(don't|didn't)" accepts either (as "do not" / "did not").
Slot = tuple[tuple[str, ...], ...]


def ref_slots(text: str) -> list[Slot]:
    slots: list[Slot] = []
    for token in re.findall(r"\([^)]*\)|[^\s()]+", text):
        if token.startswith("("):
            options = {tuple(norm(o)) for o in token[1:-1].split("|")}
            if "|" not in token:
                options.add(())  # a single bracketed word is optional
            if any(options):
                slots.append(tuple(sorted(options)))
        else:
            slots.extend(((w,),) for w in norm(token))
    return slots


def edits(ref: list[Slot], hyp: list[str]) -> int:
    """Fewest word edits turning the hypothesis into some reading of the reference.

    A slot is matched by any of its sequences (an empty one makes it free to skip); a
    missing slot costs its shortest reading, a wrong word in its place costs 1.
    """
    n, m = len(ref), len(hyp)
    best = [[0] * (m + 1) for _ in range(n + 1)]
    for j in range(m + 1):
        best[n][j] = m - j
    for i in range(n - 1, -1, -1):
        shortest = min(len(o) for o in ref[i])
        for j in range(m, -1, -1):
            cost = best[i + 1][j] + shortest  # slot missing (free when it allows nothing)
            if j < m:
                cost = min(cost, best[i][j + 1] + 1)  # extra word in the hypothesis
                if shortest:
                    cost = min(cost, best[i + 1][j + 1] + 1)  # wrong word in its place
            for option in ref[i]:
                k = len(option)
                if k and tuple(hyp[j : j + k]) == option:
                    cost = min(cost, best[i + 1][j + k])
            best[i][j] = cost
    return best[0][0]


def ref_words(ref: list[Slot]) -> int:
    return sum(min(len(o) for o in slot) for slot in ref)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(show: bool) -> None:
    labels = {}
    for rec in load_jsonl(DATA / "labels.jsonl"):
        labels[(rec["session"], rec["id"])] = rec
    refs = {k: r["reference"] for k, r in labels.items() if r["reference"] and not r["exclude"]}
    silent = {k for k, r in labels.items() if r["noSpeech"] and not r["exclude"]}

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
        words = sum(ref_words(ref_slots(refs[k])) for k in keys)
        exact = sum(edits(ref_slots(refs[k]), norm(by_key[k])) == 0 for k in keys)
        print(f"{backend:28} {len(keys):5} {errs / words:6.1%} {exact:3}/{len(keys)}")
        if show:
            for k in keys:
                if edits(ref_slots(refs[k]), norm(by_key[k])):
                    print(f"    {k[1]}  ref: {refs[k]!r}\n           hyp: {by_key[k]!r}")
    report_silence(silent, hyps)


def report_silence(silent: set, hyps: dict[str, dict]) -> None:
    """On clips labelled 'no speech', any words a backend writes are invented."""
    print(f"\n{len(silent)} clips labelled no speech\n")
    print(f"{'backend':28} {'clips':>5} {'invented words on':>18}")
    for backend, by_key in sorted(hyps.items()):
        keys = [k for k in silent if k in by_key]
        if keys:
            bad = [k for k in keys if norm(by_key[k])]
            print(f"{backend:28} {len(keys):5} {len(bad):12} clips")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print every mismatch")
    main(ap.parse_args().show)
