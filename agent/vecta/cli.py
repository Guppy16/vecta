"""`vecta` — the agent's hands. Talks to a running vecta-server over HTTP.

    vecta sessions                       list live sessions
    vecta tail SID [-n 30] [-f]          show (and follow) a session's inbox
    vecta say SID "text" [--silent]      message the user (spoken unless --silent)
    vecta task SID "text"                set the task label shown on the phone
    vecta mark SID "the left button" [--label 1] [--color "#46C46A"]
    vecta unmark SID [ID]                clear one marker or all
    vecta watch SID "a hand touches the panel"   start a VLM watch (empty text stops it)
    vecta ask-capture SID "show me the back"
    vecta page SID file.html

All commands print the server's reply as JSON.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx

SERVER = os.environ.get("VECTA_SERVER", "http://127.0.0.1:8000")


def _post(path: str, body: dict) -> None:
    r = httpx.post(SERVER + path, json=body, timeout=60)
    print(
        json.dumps(
            r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
        )
    )
    if r.status_code >= 400:
        sys.exit(1)


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="vecta", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("sessions")

    t = sub.add_parser("tail")
    t.add_argument("sid")
    t.add_argument("-n", type=int, default=30)
    t.add_argument("-f", action="store_true", help="follow")

    s = sub.add_parser("say")
    s.add_argument("sid")
    s.add_argument("text")
    s.add_argument("--silent", action="store_true", help="text only, no speech")
    s.add_argument("--status", default="answer")

    tk = sub.add_parser("task")
    tk.add_argument("sid")
    tk.add_argument("text")

    mk = sub.add_parser("mark")
    mk.add_argument("sid")
    mk.add_argument("query", help="what to find in the live view, e.g. 'the leftmost button'")
    mk.add_argument("--label", default="")
    mk.add_argument("--color", default="#46C46A")
    mk.add_argument("--id", default="")

    um = sub.add_parser("unmark")
    um.add_argument("sid")
    um.add_argument("id", nargs="?", default="")

    w = sub.add_parser("watch")
    w.add_argument("sid")
    w.add_argument("text", nargs="?", default="")

    ac = sub.add_parser("ask-capture")
    ac.add_argument("sid")
    ac.add_argument("hint")
    ac.add_argument("--kind", default="photo")

    pg = sub.add_parser("page")
    pg.add_argument("sid")
    pg.add_argument("file", type=Path)

    a = ap.parse_args()
    match a.cmd:
        case "sessions":
            print(json.dumps(httpx.get(SERVER + "/sessions", timeout=10).json(), indent=1))
        case "tail":
            _tail(a.sid, a.n, a.f)
        case "say":
            _post(
                f"/sessions/{a.sid}/say",
                {"text": a.text, "speak": not a.silent, "status": a.status},
            )
        case "task":
            _post(f"/sessions/{a.sid}/task", {"text": a.text})
        case "mark":
            _post(
                f"/sessions/{a.sid}/mark",
                {"query": a.query, "label": a.label, "color": a.color, "id": a.id},
            )
        case "unmark":
            _post(f"/sessions/{a.sid}/unmark", {"id": a.id})
        case "watch":
            _post(f"/sessions/{a.sid}/watch", {"text": a.text})
        case "ask-capture":
            _post(
                f"/sessions/{a.sid}/send",
                {"type": "capture.request", "kind": a.kind, "hint": a.hint},
            )
        case "page":
            _post(f"/sessions/{a.sid}/page", {"html": a.file.read_text()})


def _tail(sid: str, n: int, follow: bool) -> None:
    seen = 0
    while True:
        events = httpx.get(
            f"{SERVER}/sessions/{sid}/inbox", params={"n": max(n, seen + 1)}, timeout=10
        ).json()
        for e in events[seen:] if seen else events[-n:]:
            print(json.dumps(e))
        seen = len(events)
        if not follow:
            return
        time.sleep(1)
        sys.stdout.flush()


if __name__ == "__main__":
    main()
