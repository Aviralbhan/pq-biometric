"""M1 presentation attack detection: MiniFASNet (Silent-Face-Anti-Spoofing), ONNX.

Two pretrained models, each on its own 80x80 patch scale (2.7 and 4.0). Input is
raw BGR 0-255 (Silent-Face's to_tensor does not divide by 255). Softmax class 1 is
"real". As in Silent-Face's test.py, the two softmax outputs are summed (averaged here).

Two scoring rules (detectors.m1_pad.decision_rule):
- "p_real":  score = 1 - mean P(real); a threshold is set from data (S11).
- "argmax":  Silent-Face's own rule, real iff class 1 is the most probable class. Encoded as
             score = 0.5 + (max(P(class 0), P(class 2)) - P(real)) / 2, so "real" <=> score < 0.5
             and the threshold is 0.5. Used while no CelebA-Spoof data exists to calibrate M1.
Class probabilities are averaged over frames before the rule is applied.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from detectors.base import Detector, Sample, load_or_stub, onnx_session, softmax

REAL_CLASS = 1


class MiniFASNetPAD(Detector):
    name = "m1_pad"

    def __init__(self, model_paths: list[Path], version: str, threads: int = 2, decision_rule: str = "p_real"):
        if decision_rule not in ("p_real", "argmax"):
            raise ValueError(f"unknown M1 decision_rule {decision_rule!r}")
        self.version = version
        self.decision_rule = decision_rule
        self._sessions = [onnx_session(p, threads) for p in model_paths]
        self._inputs = [s.get_inputs()[0].name for s in self._sessions]

    def class_probs(self, patches_per_model: list[list[np.ndarray]]) -> np.ndarray:
        """patches_per_model[m][i] = 80x80 BGR patch i for model m. Returns (n_patches, 3) mean class probabilities."""
        total = None
        for sess, inp, patches in zip(self._sessions, self._inputs, patches_per_model):
            batch = np.stack(patches).astype(np.float32).transpose(0, 3, 1, 2).copy()
            probs = softmax(sess.run(None, {inp: batch})[0])
            total = probs if total is None else total + probs
        return total / len(self._sessions)

    def p_real(self, patches_per_model: list[list[np.ndarray]]) -> np.ndarray:
        return self.class_probs(patches_per_model)[:, REAL_CLASS]

    def score_probs(self, probs: np.ndarray) -> float:
        """Spoof score from (frame-averaged) class probabilities."""
        if self.decision_rule == "argmax":
            spoof = max(p for i, p in enumerate(probs) if i != REAL_CLASS)
            return float(0.5 + (spoof - probs[REAL_CLASS]) / 2)
        return float(1.0 - probs[REAL_CLASS])

    def score(self, sample: Sample) -> float:
        if not sample.crops:
            return 0.0
        per_model = [[c.pad80[m] for c in sample.crops] for m in range(len(self._sessions))]
        return self.score_probs(self.class_probs(per_model).mean(axis=0))


def build(config: dict[str, Any]) -> Detector:
    c = config["detectors"]["m1_pad"]
    if len(c["models"]) != len(config["preprocessing"]["pad_scales"]):
        raise ValueError("detectors.m1_pad.models must match preprocessing.pad_scales one to one")
    return load_or_stub(
        "m1_pad", c["models"],
        lambda ps: MiniFASNetPAD(ps, c["version"], config["detectors"]["onnx_threads"], c.get("decision_rule", "p_real")), config
    )
