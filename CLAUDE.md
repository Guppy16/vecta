# Vecta — working notes for Claude

Akash is the package maintainer and reviewer; Claude does most of the coding.
Optimise for code Akash can read and maintain.

## Layout
- `android/` — Kotlin/Compose app. Thin client: camera + webview + one input bar.
- `agent/` — Python package `vecta` (`vecta.server`, `vecta.jobs`, `vecta.protocol`).
  Runs on the Strix Halo box (`guppy`, Tailscale). `scripts/legacy/` is unlinted history.
- `strix-history` branch — the box's original git history, read-only.

## Python
- `uv` for everything (`uv sync`, `uv run`); package installed editable.
- `ruff` for lint, format and import sorting — run `uv run ruff check --fix && uv run ruff format` before committing.
- **Absolute imports only** (`from vecta.jobs.yolo import ...`). Never relative.
- Python 3.14: annotations are lazy, so no `from __future__ import annotations`.
- Prefer `asyncio`, `dataclasses`, type hints. Comment the why, not the what; don't over-comment.

## Kotlin
- Build from the CLI (`./gradlew`), Akash reads in Android Studio.

## Infra rules
- The Strix box runs other projects (Home Assistant :8123, wyoming :10300, Lemonade :13305).
  **Ask before** installing, starting services, using the GPU or binding ports there.
- adb / phone commands: propose them, let Akash run them first.
- `.env` holds secrets (`GH_PAT_TOKEN`). Never read or print it.

## Git
- Small PRs per feature; Akash reviews. Commit messages: `type: summary` (feat/fix/chore/docs).
