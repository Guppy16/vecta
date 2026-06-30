"""
Vectaar — Hand-State Server (consolidated)
==========================================
Per frame, classifies the hand into: no_hand / hand / grasp / point.

This is the end state of the iteration we did:
  - MediaPipe Hand Landmarker for 21 landmarks
  - a constant-velocity Kalman filter on the landmarks: smooths jitter AND
    carries the hand through brief detection dropouts (slow hand, short gap);
    its covariance decides when to give up and call no_hand
  - grasp = "object in grip": non-skin fraction inside the grip polygon
    (palm + fingertips). Colour-agnostic (works on white/clear objects).
  - point = index extended (foreshortening-robust straightness) AND the index
    tip isolated from the other fingertips AND nothing in the grip.

Held-out accuracy (tested on a clip it was NOT tuned on): ~60% strict, 4-class.
Honest ceilings (don't be surprised by these in testing):
  - grasp drops when the hand is fully enveloped/occluded by the object for
    seconds at a time — the hand is invisible, no tracker recovers it. The fix
    is an OBJECT detector (next increment), not more hand work.
  - the skin model (SKIN_* below) is tuned to one person/lighting; other users
    or lighting will need it re-fit, or swapped for an adaptive skin model.

Wire protocol (matches the Android client): each binary ws message is
8-byte big-endian client timestamp (ms) followed by JPEG bytes. We reply with
a JSON state object per processed frame. Latest-frame-wins on the receive side.

Run:
  pip install fastapi uvicorn opencv-python mediapipe numpy
  # model bundle (one-time):
  #   https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
  #   -> assets/models/hand_landmarker.task
  uvicorn vectaar_hand_server:app --host 0.0.0.0 --port 8000

The classifier (HandStateClassifier) is independent of the server — that's the
piece you'd re-implement on-device later; the FastAPI part is just transport.
"""

import sys
import json
import asyncio
from collections import deque, Counter

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_PATH = "assets/models/hand_landmarker.task"

# ---- calibrated constants (tuned on our two test clips) ----
GRASP_NONSKIN = 0.09     # grip non-skin fraction above this => grasp
PT_STRAIGHT   = 0.85     # index straightness (2D chord/path) above this => extended
PT_TIPISO     = 0.35     # index-tip isolation from other tips (/hand size) => pointing
SMOOTH_FRAMES = 5        # causal majority-vote window (kills 1-2 frame flicker)
# static skin model (YCrCb). Person/lighting specific — re-fit per user if needed.
SKIN_CR = (133, 173)
SKIN_CB = (77, 127)

# landmark indices: (mcp, pip, dip, tip) per finger
FINGERS = {"index":(5,6,7,8), "middle":(9,10,11,12), "ring":(13,14,15,16), "pinky":(17,18,19,20)}


# ======================================================================
#  Kalman filter over the 21 landmarks (independent 2D constant-velocity)
# ======================================================================
class KalmanHand:
    """One constant-velocity filter per landmark coordinate, batched.
    step(meas) returns (filtered_xy[21,2] or None, active, predicted)."""
    def __init__(self, sigma_a=0.03, r_var=1e-4, lost_var=0.025):
        self.F = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], float)
        self.H = np.array([[1,0,0,0],[0,1,0,0]], float)
        q = sigma_a**2
        self.Q = q*np.array([[.25,0,.5,0],[0,.25,0,.5],[.5,0,1,0],[0,.5,0,1]])
        self.R = r_var*np.eye(2)
        self.lost_var = lost_var
        self.X = None; self.P = None; self.init = False

    def step(self, meas):
        I4 = np.eye(4)
        if self.init:                                  # predict
            self.X = self.X @ self.F.T
            self.P = np.einsum('ij,njk,lk->nil', self.F, self.P, self.F) + self.Q
        if meas is not None:                           # update / (re)init
            if not self.init:
                self.X = np.concatenate([meas, np.zeros((21,2))], 1)
                self.P = np.tile(I4*0.01, (21,1,1)); self.init = True
            else:
                y = meas - self.X @ self.H.T
                S = np.einsum('ij,njk,lk->nil', self.H, self.P, self.H) + self.R
                K = np.einsum('nji,nik->njk', np.einsum('njk,ik->nji', self.P, self.H),
                              np.linalg.inv(S))
                self.X = self.X + np.einsum('njk,nk->nj', K, y)
                KH = np.einsum('nik,kj->nij', K, self.H)
                self.P = np.einsum('nij,njk->nik', I4 - KH, self.P)
        if not self.init:
            return None, False, False
        posvar = float(np.mean(self.P[:,0,0] + self.P[:,1,1]))
        if posvar > self.lost_var:                     # too uncertain -> give up
            self.init = False
            return None, False, False
        return self.X[:, :2].copy(), True, (meas is None)


# ======================================================================
#  Features
# ======================================================================
def grip_nonskin(frame_bgr, lm_norm):
    """Fraction of non-skin pixels inside the grip polygon (palm + fingertips).
    High => something (an object) is occupying the grip. Colour-agnostic."""
    h, w = frame_bgr.shape[:2]
    P = (lm_norm * [w, h]).astype(np.int32)
    poly = cv2.convexHull(P[[0,4,8,12,16,20]])
    mask = np.zeros((h, w), np.uint8); cv2.fillConvexPoly(mask, poly, 255); m = mask > 0
    tot = int(m.sum())
    if tot < 50:
        return 0.0
    ycc = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YCrCb)
    cr, cb = ycc[:,:,1], ycc[:,:,2]
    skin = (cr>=SKIN_CR[0])&(cr<=SKIN_CR[1])&(cb>=SKIN_CB[0])&(cb<=SKIN_CB[1])
    return float((m & ~skin).sum() / tot)

def _straight2d(lm, mcp, pip, dip, tip):
    a = lm[[mcp, pip, dip, tip]]
    chord = np.linalg.norm(a[3]-a[0])
    path = sum(np.linalg.norm(a[i+1]-a[i]) for i in range(3))
    return chord / max(path, 1e-6)

def index_straight(lm):
    return _straight2d(lm, *FINGERS["index"])

def tip_isolation(lm):
    """Index-tip distance from the centroid of the other 3 tips, / hand size.
    The real signature of pointing (index sticks out from the pack)."""
    others = lm[[12,16,20]].mean(0)
    hs = np.linalg.norm(lm.max(0) - lm.min(0))
    return float(np.linalg.norm(lm[8] - others) / max(hs, 1e-6))


# ======================================================================
#  Classifier  (this is the portable piece — reimplement on-device later)
# ======================================================================
class HandStateClassifier:
    def __init__(self, model_path=MODEL_PATH):
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=base, running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1, min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5, min_tracking_confidence=0.5)
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)
        self.kf = KalmanHand()
        self.hist = deque(maxlen=SMOOTH_FRAMES)
        self.last_ts = 0

    def _raw_state(self, frame_bgr, lm):
        obj = grip_nonskin(frame_bgr, lm)
        if (index_straight(lm) > PT_STRAIGHT and tip_isolation(lm) > PT_TIPISO
                and obj < GRASP_NONSKIN):
            return "point", obj
        if obj > GRASP_NONSKIN:
            return "grasp", obj
        return "hand", obj

    def process(self, frame_bgr, ts_ms):
        """Returns the per-frame state dict. ts_ms must be non-decreasing."""
        ts_ms = max(int(ts_ms), self.last_ts + 1); self.last_ts = ts_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        meas = None
        if res.hand_landmarks:
            meas = np.array([[p.x, p.y] for p in res.hand_landmarks[0]], float)
        lm, active, predicted = self.kf.step(meas)

        if not active:
            raw, obj = "no_hand", None
        else:
            raw, obj = self._raw_state(frame_bgr, lm)

        self.hist.append(raw)
        state = Counter(self.hist).most_common(1)[0][0]   # causal majority vote
        return {
            "state": state,
            "raw_state": raw,
            "object_score": obj,
            "detected": meas is not None,   # MediaPipe saw the hand this frame
            "predicted": bool(predicted),   # hand carried by Kalman through a gap
            "landmarks": lm.tolist() if lm is not None else None,  # normalized [21,2]
        }

    def close(self):
        self.landmarker.close()


# ======================================================================
#  FastAPI websocket transport (matches the Android wire protocol)
# ======================================================================
app = FastAPI()

@app.get("/ping")
async def ping():
    return {"status": "ok", "message": "Vectaar hand-state server reachable"}

@app.websocket("/ws")
async def ws(websocket: WebSocket):
    await websocket.accept()
    print("[+] Client connected.")
    clf = HandStateClassifier()
    latest = {"data": None}
    new_frame = asyncio.Event(); disconnected = asyncio.Event()

    async def receiver():
        try:
            while True:
                latest["data"] = await websocket.receive_bytes(); new_frame.set()
        except WebSocketDisconnect:
            pass
        finally:
            disconnected.set(); new_frame.set()

    recv = asyncio.create_task(receiver())
    try:
        while not disconnected.is_set():
            await new_frame.wait(); new_frame.clear()
            data = latest["data"]; latest["data"] = None
            if data is None or len(data) < 8:
                continue
            ts = int.from_bytes(data[:8], "big")
            img = cv2.imdecode(np.frombuffer(data[8:], np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
            # inference is blocking; keep the event loop free
            out = await asyncio.to_thread(clf.process, img, ts)
            tag = "KF" if out["predicted"] else ("  " if out["detected"] else "--")
            objs = f"{out['object_score']:.2f}" if out["object_score"] is not None else " -- "
            sys.stdout.write(f"\r[{out['state']:<7}] {tag} obj={objs}      "); sys.stdout.flush()
            out["timestamp"] = ts
            await websocket.send_text(json.dumps(out))
    except WebSocketDisconnect:
        print("\n[-] Client disconnected.")
    finally:
        recv.cancel(); clf.close(); print("\n[-] Session closed.")