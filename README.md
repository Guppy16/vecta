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

## How it fits together

```
phone (android/)                          Strix box (agent/)
  camera ──H.264──▶ WebRTC video track ──▶ FrameBuffer (last ~3 s)
  task bar / shutter / page taps ──▶ data channel "control" ◀── page.render, status, capture.request
  WebView ◀── HTML (+ assets over HTTP: /assets/<session>/<file>)
```

Signalling is one `POST /rtc/offer`; no STUN/TURN — everything is on the tailnet.
Message shapes live in `agent/vecta/protocol/messages.py` and are mirrored in
`android/.../transport/Messages.kt`.

## Agent server

```sh
cd agent
uv sync                 # base deps
uv sync --extra gpu     # + torch (ROCm 7.2 wheels) / ultralytics / mediapipe — on the box only
uv run vecta-server     # VECTA_PORT (8000), VECTA_HOST, VECTA_DATA_DIR (~/.vecta)
uv run pytest
```

Test the whole loop without a phone: `uv run python scripts/fake_phone.py --video clip.mp4`
streams a file to the server and prints what comes back.

## Android

`android/local.properties` needs `SERVER_IP="100.x.x.x:8000"` (the box's Tailscale address).

```sh
cd android
./gradlew installDebug
```
