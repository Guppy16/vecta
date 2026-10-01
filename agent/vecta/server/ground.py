"""Live markers: keep a description like "the leftmost button" pinned to the
thing in the camera view, frame after frame, so the phone can draw it.

v1 grounds with Qwen3-VL through Lemonade (it answers with a box in a
0..1000 grid), about one update per second. moondream's `point`/`detect`
will replace it for ~10 Hz once it is in-process.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import re
import time
from dataclasses import dataclass, field

from PIL import Image

from vecta.protocol import messages as m
from vecta.server.sessions import Session
from vecta.server.vlm import Vlm

log = logging.getLogger(__name__)

GROUND_WIDTH = 640  # px sent to the model; boxes come back on a 0..1000 grid regardless
INTERVAL_S = 0.3

GROUND_SYSTEM = (
    "You locate one thing in a camera frame. Answer with ONLY JSON: "
    '{"found": true|false, "bbox_2d": [x1, y1, x2, y2]} with coordinates on a 0..1000 grid '
    "(x from the left edge, y from the top). If it is not visible, found=false and no bbox."
)


@dataclass
class MarkerSpec:
    id: str
    query: str
    label: str = ""
    color: str = "#46C46A"
    # last known normalised box (cx, cy, w, h); None until first found
    box: tuple[float, float, float, float] | None = None
    last_seen: float = 0.0
    misses: int = 0


@dataclass
class MarkerTracker:
    session: Session
    vlm: Vlm
    send: object  # Callable[[m.Message], None]
    markers: dict[str, MarkerSpec] = field(default_factory=dict)
    _task: asyncio.Task[None] | None = None

    def set(self, spec: MarkerSpec) -> None:
        self.markers[spec.id] = spec
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name=f"ground-{self.session.id}")

    def remove(self, marker_id: str | None) -> None:
        if marker_id:
            self.markers.pop(marker_id, None)
        else:
            self.markers.clear()
        self._push()
        if not self.markers and self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self) -> None:
        while self.markers:
            t0 = time.monotonic()
            frame = self.session.frames.latest()
            if frame is not None:
                jpeg = await asyncio.to_thread(_small_jpeg, frame.image.to_image())
                for spec in list(self.markers.values()):
                    await self._locate(spec, jpeg)
                self._push()
            await asyncio.sleep(max(0.0, INTERVAL_S - (time.monotonic() - t0)))

    async def _locate(self, spec: MarkerSpec, jpeg: bytes) -> None:
        try:
            text = await self.vlm.chat(
                f"{GROUND_SYSTEM}", f"Locate: {spec.query}", [jpeg], 60, True
            )
        except Exception as e:
            log.warning("ground error: %s", e)
            return
        box = parse_box(text)
        if box is None:
            spec.misses += 1
            if spec.misses >= 3:
                spec.box = None  # fade on the phone
            return
        spec.misses, spec.last_seen, spec.box = 0, time.time(), box

    def _push(self) -> None:
        latest = self.session.frames.latest()
        w, h = latest.size if latest else (0, 0)
        self.send(
            m.OverlaySet(
                markers=[
                    {
                        "id": s.id,
                        "label": s.label,
                        "color": s.color,
                        "x": s.box[0],
                        "y": s.box[1],
                        "w": s.box[2],
                        "h": s.box[3],
                    }
                    for s in self.markers.values()
                    if s.box is not None
                ],
                frame_w=w,
                frame_h=h,
            )
        )


def _small_jpeg(img: Image.Image) -> bytes:
    small = img.resize((GROUND_WIDTH, int(img.height * GROUND_WIDTH / img.width)))
    buf = io.BytesIO()
    small.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def parse_box(text: str) -> tuple[float, float, float, float] | None:
    """Qwen-style {"found":.., "bbox_2d":[x1,y1,x2,y2]} (0..1000 grid) -> (cx, cy, w, h) in 0..1."""
    mo = re.search(r"\{.*\}", text, re.S)
    if not mo:
        return None
    try:
        obj = json.loads(mo.group(0))
    except json.JSONDecodeError:
        return None
    bbox = obj.get("bbox_2d") or obj.get("bbox")
    if not obj.get("found", True) or not (isinstance(bbox, list) and len(bbox) == 4):
        return None
    x1, y1, x2, y2 = (float(v) / 1000 for v in bbox)
    if x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)
