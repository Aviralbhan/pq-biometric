"""One face detection per frame, then every crop the detectors need.

- 112x112 ArcFace crop, aligned on 5 landmarks (matcher, M2)
- 224x224 square crop around the face box (M3 GenD)
- 80x80 patches at the MiniFASNet scales, using Silent-Face's CropImage rule (M1)

Detection is YuNet (OpenCV FaceDetectorYN). If the weights are missing and stubs
are allowed, a centre-box stub is used so the protocol can run end to end; its
version is "stub" and the verdict records that.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

import cv2
import numpy as np

from settings import resolve

log = logging.getLogger(__name__)

# InsightFace's 112x112 ArcFace landmark template: eye (image left), eye (image right),
# nose tip, mouth corner (image left), mouth corner (image right). YuNet uses the same order.
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x, y, w, h in pixels
    landmarks: np.ndarray  # (5, 2)
    score: float

    @property
    def width(self) -> float:
        return self.box[2]


@dataclass
class FaceCrops:
    face: Face
    arcface112: np.ndarray
    gend224: np.ndarray
    pad80: list[np.ndarray] = field(default_factory=list)  # one per configured scale


class FaceDetector(Protocol):
    version: str

    def detect(self, img: np.ndarray) -> list[Face]: ...


class YuNetDetector:
    def __init__(self, model_path: str, score_threshold: float = 0.8, nms_threshold: float = 0.3, top_k: int = 50):
        self.version = "yunet-2023mar"
        self._det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, nms_threshold, top_k)

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        self._det.setInputSize((w, h))
        _, rows = self._det.detect(img)
        if rows is None:
            return []
        faces = [
            Face(box=tuple(float(v) for v in r[:4]), landmarks=r[4:14].reshape(5, 2).astype(np.float32), score=float(r[14]))
            for r in rows
        ]
        return sorted(faces, key=lambda f: (f.score, f.box[2] * f.box[3]), reverse=True)  # most confident first


class CenterFaceStub:
    """Pretends a face fills the centre of the frame. Only for running without weights."""

    version = "stub"

    def detect(self, img: np.ndarray) -> list[Face]:
        h, w = img.shape[:2]
        side = 0.5 * min(h, w)
        x, y = (w - side) / 2, (h - side) / 2
        landmarks = ARCFACE_TEMPLATE / 112.0 * side + np.array([x, y], dtype=np.float32)
        return [Face(box=(x, y, side, side), landmarks=landmarks, score=1.0)]


def align_arcface(img: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    template = ARCFACE_TEMPLATE * (size / 112.0)
    matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), template, method=cv2.LMEDS)
    if matrix is None:
        raise ValueError("could not estimate alignment")
    return cv2.warpAffine(img, matrix, (size, size), borderValue=0.0)


def crop_scaled(img: np.ndarray, box: tuple[float, float, float, float], scale: float, out_w: int, out_h: int) -> np.ndarray:
    """Silent-Face-Anti-Spoofing CropImage: enlarge the box by `scale`, shift it inside the image, resize."""
    src_h, src_w = img.shape[:2]
    x, y, bw, bh = box
    scale = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * scale, bh * scale
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0, x1, y1 = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > src_w - 1:
        x0 -= x1 - src_w + 1
        x1 = src_w - 1
    if y1 > src_h - 1:
        y0 -= y1 - src_h + 1
        y1 = src_h - 1
    x0, y0, x1, y1 = int(x0), int(y0), int(x1), int(y1)
    return cv2.resize(img[y0 : y1 + 1, x0 : x1 + 1], (out_w, out_h))


def crop_square(img: np.ndarray, box: tuple[float, float, float, float], scale: float, size: int) -> np.ndarray:
    """Square crop of side max(w, h) * scale centred on the box; zero padding outside the image."""
    x, y, bw, bh = box
    side = max(bw, bh) * scale
    cx, cy = x + bw / 2, y + bh / 2
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    s = max(int(round(side)), 1)
    h, w = img.shape[:2]
    pad = max(0, -x0, -y0, x0 + s - w, y0 + s - h)
    if pad:
        img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
        x0, y0 = x0 + pad, y0 + pad
    return cv2.resize(img[y0 : y0 + s, x0 : x0 + s], (size, size), interpolation=cv2.INTER_AREA)


class FacePreprocessor:
    def __init__(
        self,
        detector: FaceDetector,
        min_face_px: int = 112,
        gend_box_scale: float = 1.3,
        pad_scales: tuple[float, ...] = (2.7, 4.0),
        score_margin: float = 0.1,
    ):
        self.detector = detector
        self.score_margin = score_margin
        self.min_face_px = min_face_px
        self.gend_box_scale = gend_box_scale
        self.pad_scales = tuple(pad_scales)

    @property
    def version(self) -> str:
        return self.detector.version

    def best_face(self, img: np.ndarray) -> Face | None:
        """Among faces >= min_face_px whose score is within score_margin of the best, the largest.

        Neither "largest" nor "most confident" alone is safe on FRLL at YuNet score 0.5: a weak false
        box below the chin can be larger than the face (030: 0.65 vs 0.88), and a small false box next
        to a 3/4 face can be slightly more confident than the face (114: 0.837 vs 0.811).
        """
        faces = [f for f in self.detector.detect(img) if f.width >= self.min_face_px]
        if not faces:
            return None
        top = max(f.score for f in faces)
        return max((f for f in faces if f.score >= top - self.score_margin), key=lambda f: f.box[2] * f.box[3])

    def crops(self, img: np.ndarray, face: Face) -> FaceCrops:
        return FaceCrops(
            face=face,
            arcface112=align_arcface(img, face.landmarks, 112),
            gend224=crop_square(img, face.box, self.gend_box_scale, 224),
            pad80=[crop_scaled(img, face.box, s, 80, 80) for s in self.pad_scales],
        )

    def process(self, img: np.ndarray) -> FaceCrops | None:
        face = self.best_face(img)
        return self.crops(img, face) if face else None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> FacePreprocessor:
        p = config["preprocessing"]
        path = resolve(p["yunet_model"])
        if path.exists():
            detector: FaceDetector = YuNetDetector(str(path), p["score_threshold"])
        elif config["detectors"].get("allow_stubs"):
            log.warning("YuNet weights missing at %s; using centre-box stub detector", path)
            detector = CenterFaceStub()
        else:
            raise FileNotFoundError(path)
        return cls(detector, p["min_face_px"], p["gend_box_scale"], tuple(p["pad_scales"]), p.get("face_score_margin", 0.1))
