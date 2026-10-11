"""M3 deepfake detection: GenD (CLIP ViT-L/14 + LinearNorm head), exported to ONNX.

Preprocessing matches CLIPProcessor for a 224x224 input: RGB, /255, CLIP mean/std.
(The face crop is already 224x224, so CLIPProcessor's resize + centre crop is a no-op.)
The fake class index is checked at export time (scripts/export_onnx.py) and stored
in the config. Frame scores are averaged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from detectors.base import Detector, Sample, load_or_stub, onnx_session, softmax
from preprocessing.video import sample_indices

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    batch = np.stack([c[:, :, ::-1] for c in crops_bgr]).astype(np.float32) / 255.0
    return ((batch - CLIP_MEAN) / CLIP_STD).transpose(0, 3, 1, 2).copy()


class GenDDeepfake(Detector):
    name = "m3_deepfake"

    def __init__(self, model_path: Path, version: str, fake_index: int = 1, threads: int = 2, max_frames: int | None = None):
        self.version = version
        self.fake_index = fake_index
        self.max_frames = max_frames  # CLIP ViT-L/14 costs ~0.2-0.6 s per frame on CPU
        self._sess = onnx_session(model_path, threads)
        self._input = self._sess.get_inputs()[0].name

    def p_fake(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        logits = self._sess.run(None, {self._input: preprocess(crops_bgr)})[0]
        return softmax(logits)[:, self.fake_index]

    def score(self, sample: Sample) -> float:
        if not sample.crops:
            return 0.0
        crops = sample.crops
        if self.max_frames:
            crops = [crops[i] for i in sample_indices(len(crops), self.max_frames)]
        return float(self.p_fake([c.gend224 for c in crops]).mean())


def build(config: dict[str, Any]) -> Detector:
    c = config["detectors"]["m3_deepfake"]
    return load_or_stub(
        "m3_deepfake",
        [c["model"]],
        lambda ps: GenDDeepfake(ps[0], c["version"], c.get("fake_index", 1), c.get("threads", config["detectors"]["onnx_threads"]), c.get("max_frames")),
        config,
    )
