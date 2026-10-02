"""Score the talker's turn-taking decisions on a labelled calibration set.

Each case is one real utterance with the dialogue and briefing the talker had at the
time, and the actions that would have been right: answer, look (camera), escalate (to
the main agent) or silent. The talker's choice is scored raw (the prompt alone) and with
the server's keyword backstops, so a better prompt shows up as the raw score rising.

    uv run --no-sync python scripts/talker_eval.py                      # current PERSONA
    uv run --no-sync python scripts/talker_eval.py --persona new.txt    # try a candidate

The set lives in data/calibration/talker_turns.jsonl (gitignored: it is real speech).
"""

import argparse
import asyncio
import json
from pathlib import Path

from vecta.server import talker as talker_mod
from vecta.server.talker import HOWTO_HINTS, INSTRUCTION_HINTS, LOOK_HINTS, Talker

DEFAULT_SET = Path(__file__).resolve().parents[2] / "data" / "calibration" / "talker_turns.jsonl"


def action(reply: talker_mod.Reply) -> str:
    if reply.tool == "look":
        return "look"
    if reply.escalate:
        return "escalate"
    return "answer" if reply.text else "silent"


def with_backstops(utterance: str, raw: str, reply: talker_mod.Reply) -> str:
    """What the server would have done (see app._talk)."""
    if HOWTO_HINTS.search(utterance):
        return "escalate"
    if LOOK_HINTS.search(utterance):
        return "look"
    if raw == "answer" and INSTRUCTION_HINTS.search(reply.text):
        return "escalate"
    return raw


async def main(cases: list[dict], persona: str | None) -> None:
    if persona:
        talker_mod.PERSONA = persona
    raw_ok = backed_ok = 0
    for case in cases:
        t = Talker(briefing=case["briefing"])
        t.history = list(case["history"])
        reply = await t.turn(case["utterance"])
        raw = action(reply)
        backed = with_backstops(case["utterance"], raw, reply)
        raw_ok += raw in case["allowed"]
        backed_ok += backed in case["allowed"]
        if raw not in case["allowed"]:
            print(
                f"✗ {case['id']} want {'/'.join(case['allowed']):16} got {raw:8} "
                f"(backstops: {backed:8}) {case['utterance'][:60]!r} -> {reply.text[:60]!r}"
            )
    n = len(cases)
    print(
        f"raw prompt: {raw_ok}/{n} ({raw_ok / n:.0%})   with backstops: {backed_ok}/{n} "
        f"({backed_ok / n:.0%})"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", type=Path, default=DEFAULT_SET)
    ap.add_argument("--persona", type=Path, help="text file with a candidate PERSONA")
    args = ap.parse_args()
    cases = [json.loads(line) for line in args.set.read_text().splitlines() if line.strip()]
    asyncio.run(main(cases, args.persona.read_text().strip() if args.persona else None))
