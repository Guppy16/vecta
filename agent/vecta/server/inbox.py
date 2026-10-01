"""The session inbox: an append-only event log the agent watches.

Everything that reaches the server from the phone (task, text, voice, captures,
distinct keyframes, taps) is appended as one JSON line to
``<session dir>/inbox.jsonl``. An agent — a Claude Code session on the box, or
a developer over ssh — tails that file and replies through the `vecta` CLI.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class Inbox:
    def __init__(self, path: Path) -> None:
        self.path = path

    def append(self, kind: str, **fields: Any) -> dict[str, Any]:
        event = {"ts": round(time.time(), 3), "type": kind, **fields}
        with self.path.open("a") as f:
            f.write(json.dumps(event) + "\n")
        return event

    def tail(self, n: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text().splitlines()[-n:]
        return [json.loads(line) for line in lines if line.strip()]
