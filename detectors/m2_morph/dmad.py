"""M2 morphing attack detection: differential (D-MAD) on ArcFace embeddings.

Feature = ArcFace(enrolled reference) - ArcFace(live probe), classified by an RBF
SVM. A morphed enrolment photo matches the live subject only partly, so the
difference vector has a different distribution from a genuine reference-probe pair.

Training pairs (scripts/train_m2.py, FRLL + FRLL-Morphs, 5 identity-grouped folds):
  label 1: reference = morph of subjects A and B, probe = bona fide image of A or B
  label 0: reference = bona fide image of A,      probe = another bona fide image of A

Score = logistic(SVM decision value), in (0, 1). The SVM is not probability-calibrated:
sklearn's internal calibration CV is not grouped by identity and would leak subjects.
The threshold is set on held-out scores instead (BPCER 0.5%, see CLAUDE.md).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detectors.base import Detector, Sample, load_or_stub


def dmad_feature(reference: np.ndarray, probe: np.ndarray) -> np.ndarray:
    return (np.asarray(reference, np.float32) - np.asarray(probe, np.float32)).reshape(-1)


def train_svm(features: np.ndarray, labels: np.ndarray, c: float = 1.0, gamma: str | float = "scale", seed: int = 0) -> Pipeline:
    model = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=c, gamma=gamma, class_weight="balanced", random_state=seed))
    model.fit(features, labels)
    return model


def svm_score(model: Pipeline, features: np.ndarray) -> np.ndarray:
    """Morph score in (0, 1): logistic of the SVM decision value (monotonic, so thresholds are unaffected)."""
    return 1.0 / (1.0 + np.exp(-model.decision_function(np.atleast_2d(features))))


def save(model: Pipeline, path: str | Path, version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "version": version}, path)


class DMADMorph(Detector):
    name = "m2_morph"

    def __init__(self, path: Path):
        bundle = joblib.load(path)
        self.model: Pipeline = bundle["model"]
        self.version = bundle["version"]

    def score_pair(self, reference: np.ndarray, probe: np.ndarray) -> float:
        return float(svm_score(self.model, dmad_feature(reference, probe))[0])

    def score(self, sample: Sample) -> float:
        if sample.template is None or sample.probe_embedding is None:
            return 0.0
        return self.score_pair(sample.template, sample.probe_embedding)


def build(config: dict[str, Any]) -> Detector:
    return load_or_stub("m2_morph", [config["detectors"]["m2_morph"]["model"]], lambda ps: DMADMorph(ps[0]), config)
