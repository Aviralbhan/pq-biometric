"""M1 presentation attack detection: MiniFASNet (Silent-Face-Anti-Spoofing), ONNX.

Two pretrained models, each on its own 80x80 patch scale (2.7 and 4.0). Input is
raw BGR 0-255 (Silent-Face's to_tensor does not divide by 255). Softmax class 1 is
"real". As in Silent-Face's test.py, the two softmax outputs are summed; the spoof
score is 1 - mean P(real). Frame scores are averaged.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from detectors.base import Detector, Sample, load_or_stub, onnx_session, softmax

REAL_CLASS = 1


class MiniFASNetPAD(Detector):
    name = "m1_pad"

    def __init__(self, model_paths: list[Path], version: str, threads: int = 2):
        self.version = version
        self._sessions = [onnx_session(p, threads) for p in model_paths]
        self._inputs = [s.get_inputs()[0].name for s in self._sessions]

    def p_real(self, patches_per_model: list[list[np.ndarray]]) -> np.ndarray:
        """patches_per_model[m][i] = 80x80 BGR patch i for model m. Returns mean P(real) per patch."""
        total = None
        for sess, inp, patches in zip(self._sessions, self._inputs, patches_per_model):
            batch = np.stack(patches).astype(np.float32).transpose(0, 3, 1, 2).copy()
            probs = softmax(sess.run(None, {inp: batch})[0])[:, REAL_CLASS]
            total = probs if total is None else total + probs
        return total / len(self._sessions)

    def score(self, sample: Sample) -> float:
        if not sample.crops:
            return 0.0
        per_model = [[c.pad80[m] for c in sample.crops] for m in range(len(self._sessions))]
        return float(1.0 - self.p_real(per_model).mean())


def build(config: dict[str, Any]) -> Detector:
    c = config["detectors"]["m1_pad"]
    if len(c["models"]) != len(config["preprocessing"]["pad_scales"]):
        raise ValueError("detectors.m1_pad.models must match preprocessing.pad_scales one to one")
    return load_or_stub(
        "m1_pad", c["models"], lambda ps: MiniFASNetPAD(ps, c["version"], config["detectors"]["onnx_threads"]), config
    )
