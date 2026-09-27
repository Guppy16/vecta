# Vecta

Point your phone at something, say what you want, get a purpose-built page back.

A task-driven vision agent: the phone captures photos/video and sends them over
WebRTC to a server that runs a Claude Agent SDK session per task. The agent
uses local GPU jobs (detection, OCR, stitching) for precise work and renders
its answer as an HTML page the phone displays — a product comparison, a
stitched control panel with numbered overlays, a count — whatever the task
needs. No chat transcript; the page *is* the interface.

## Layout

| path | what |
|---|---|
| `android/` | Kotlin/Compose client |
| `agent/` | Python package `vecta` — agent host, media ingest, GPU jobs |
| `CLAUDE.md` | conventions and infra rules |

## Server

```sh
cd agent
uv sync            # base deps
uv sync --extra gpu  # + torch/ultralytics/mediapipe (ROCm index: see pyproject)
uv run uvicorn vecta.server.agent_server:app --host 0.0.0.0 --port 8000   # legacy WS server, until v1 lands
```

## Android

```sh
cd android
./gradlew installDebug
```
