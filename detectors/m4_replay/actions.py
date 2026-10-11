"""M4 replay: confirm the challenge's two liveness actions appear in the video, in order.

Checks 1-6 (server/checks.py) already reject replayed or re-targeted bundles; this
module stops a pre-recorded video, which cannot follow actions chosen after it was
recorded. MediaPipe FaceLandmarker gives per-frame eye-blink blendshapes and a
head pose matrix; blinks and turns are turned into timed events, then matched
against the action list in order.

Score: 0.0 if the actions were seen in order, 1.0 otherwise.
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from detectors.base import Detector, Sample, load_or_stub
from preprocessing.video import decode_jpeg

ACTIONS = ("blink_twice", "turn_left", "turn_right")


@dataclass(frozen=True)
class FrameSignal:
    t_ms: float
    blink: float | None  # mean of eyeBlinkLeft/Right, None if no face
    yaw: float | None  # degrees, sign-corrected so positive = subject turned left


@dataclass(frozen=True)
class Event:
    kind: str  # "blink" | "turn_left" | "turn_right"
    t_ms: float


def detect_events(signals: list[FrameSignal], blink_close: float, blink_open: float, yaw_degrees: float) -> list[Event]:
    events: list[Event] = []
    eyes_closed = False
    turned: str | None = None
    for s in signals:
        if s.blink is not None:
            if not eyes_closed and s.blink > blink_close:
                eyes_closed = True
            elif eyes_closed and s.blink < blink_open:
                eyes_closed = False
                events.append(Event("blink", s.t_ms))
        if s.yaw is not None:
            if turned is None and abs(s.yaw) > yaw_degrees:
                turned = "turn_left" if s.yaw > 0 else "turn_right"
                events.append(Event(turned, s.t_ms))
            elif turned is not None and abs(s.yaw) < yaw_degrees / 2:
                turned = None  # back to neutral; the next turn is a new event
    return events


def actions_in_order(events: list[Event], actions: list[str]) -> bool:
    """Each action must complete strictly after the previous one completed."""
    t = -math.inf
    for action in actions:
        if action == "blink_twice":
            blinks = [e.t_ms for e in events if e.kind == "blink" and e.t_ms > t]
            if len(blinks) < 2:
                return False
            t = blinks[1]
        elif action in ("turn_left", "turn_right"):
            turns = [e.t_ms for e in events if e.kind == action and e.t_ms > t]
            if not turns:
                return False
            t = turns[0]
        else:
            raise ValueError(f"unknown action {action!r}")
    return True


def yaw_from_matrix(matrix: np.ndarray) -> float:
    """Rotation about the vertical axis, in degrees, from MediaPipe's 4x4 face transform."""
    r = np.asarray(matrix, np.float64)[:3, :3]
    return math.degrees(math.atan2(r[0, 2], r[2, 2]))


def frame_times_ms(metadata: dict[str, Any], n: int) -> list[float]:
    ts = metadata.get("frame_timestamps_ns") or []
    if len(ts) == n:
        return [(t - ts[0]) / 1e6 for t in ts]
    fps = float(metadata.get("delivered", {}).get("fps") or 30.0)
    return [i * 1000.0 / fps for i in range(n)]


def subsample(times: list[float], max_fps: float) -> list[int]:
    keep, last = [], -math.inf
    for i, t in enumerate(times):
        if t - last >= 1000.0 / max_fps - 1e-6:
            keep.append(i)
            last = t
    return keep


class ActionChecker(Detector):
    name = "m4_replay"

    def __init__(self, landmarker_path: Path, cfg: dict[str, Any]):
        from mediapipe.tasks.python import BaseOptions, vision

        self.version = cfg["version"]
        self.cfg = cfg
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(landmarker_path)),
            num_faces=1,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)
        self._lock = threading.Lock()  # one landmarker graph; calls are serialised

    def signal(self, bgr: np.ndarray, t_ms: float) -> FrameSignal:
        import mediapipe as mp

        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(bgr[:, :, ::-1]))
        with self._lock:
            result = self._landmarker.detect(image)
        if not result.face_landmarks:
            return FrameSignal(t_ms, None, None)
        shapes = {c.category_name: c.score for c in result.face_blendshapes[0]}
        blink = (shapes.get("eyeBlinkLeft", 0.0) + shapes.get("eyeBlinkRight", 0.0)) / 2
        yaw = self.cfg["yaw_sign"] * yaw_from_matrix(result.facial_transformation_matrixes[0])
        return FrameSignal(t_ms, blink, yaw)

    def signals(self, sample: Sample) -> list[FrameSignal]:
        times = frame_times_ms(sample.metadata, len(sample.jpegs))
        return [self.signal(decode_jpeg(sample.jpegs[i]), times[i]) for i in subsample(times, self.cfg["max_fps"])]

    def score(self, sample: Sample) -> float:
        events = detect_events(self.signals(sample), self.cfg["blink_close"], self.cfg["blink_open"], self.cfg["yaw_degrees"])
        return 0.0 if actions_in_order(events, sample.actions) else 1.0


def build(config: dict[str, Any]) -> Detector:
    c = config["detectors"]["m4_replay"]
    return load_or_stub("m4_replay", [c["landmarker"]], lambda ps: ActionChecker(ps[0], c), config)
