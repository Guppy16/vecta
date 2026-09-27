from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).parent

# Directory to save assets and crops
ASSETS_DIR = CURRENT_DIR / "assets"
MODELS_DIR = ASSETS_DIR / "models"
CAPTURE_DIR = ASSETS_DIR / "captured"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_WEIGHTS = MODELS_DIR / "yolov8m-world.pt"


class Predictor:
    def __init__(self, model_weights=MODEL_WEIGHTS):
        self.model = YOLO(model_weights)
        self.capture_dir = CAPTURE_DIR
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.device = 0
        self.seen_assets = set()

        self.model.set_classes(["person"])

        print("[BOOT] Running GPU warmup pass on ROCm device...")
        dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model.predict(
            dummy_frame,
            device=self.device,
            half=True,
            verbose=False,
            # tracker="bytetrack.yaml",
            # persist=True,
        )
        print("[BOOT] Server engine is hot and ready for inference!")

    def track_frame(self, frame: np.ndarray):
        """
        Execution Phase: Runs PER REQUEST / FRAME.
        Zero model-loading overhead. Pure matrix math.
        """
        # Execute tracking using your optimized ByteTrack parameters
        results = self.model.track(
            frame,
            persist=True,
            verbose=False,
            tracker="bytetrack.yaml",
            # conf=confidence,
            device=self.device,
            half=True,  # Enforce half-precision floating point math
        )
        return results

    def get_objects(self, results: list, frame: np.ndarray):
        boxes_obj = results[0].boxes
        if hasattr(boxes_obj, "id") and boxes_obj.id is not None:
            boxes = boxes_obj.xyxy.cpu().numpy().astype(int)
            track_ids = boxes_obj.id.cpu().numpy().astype(int)
            h, w = frame.shape[:2]
            for box, track_id in zip(boxes, track_ids, strict=True):
                if int(track_id) not in self.seen_assets:
                    x1, y1, x2, y2 = box
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    crop = frame[y1:y2, x1:x2]
                    if crop.size > 0:
                        out_path = CAPTURE_DIR / f"asset_id_{int(track_id)}.jpg"
                        cv2.imwrite(str(out_path), crop)
                        self.seen_assets.add(int(track_id))

                        # Print alert on a new line so it doesn't break our rolling stats line
                        print(f"\n[AI] Captured new asset — saved {out_path}")
