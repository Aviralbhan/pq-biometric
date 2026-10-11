"""M2 morphing attack detection: differential (D-MAD) on ArcFace embeddings.

Feature = ArcFace(enrolled reference) - ArcFace(live probe), classified by an RBF
SVM. A morphed enrolment photo matches the live subject only partly, so the
difference vector has a different distribution from a genuine reference-probe pair.

Training pairs (scripts/train_m2.py, FRLL + FRLL-Morphs, split by identity):
  label 1: reference = morph of subjects A and B, probe = bona fide image of A or B
  label 0: reference = bona fide image of A,      probe = another bona fide image of A
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from detectors.base import Detector, Sample, load_or_stub


def dmad_feature(reference: np.ndarray, probe: np.ndarray) -> np.ndarray:
    return (np.asarray(reference, np.float32) - np.asarray(probe, np.float32)).reshape(-1)


def train_svm(features: np.ndarray, labels: np.ndarray, c: float = 1.0, gamma: str | float = "scale", seed: int = 0) -> Pipeline:
    svm = SVC(kernel="rbf", C=c, gamma=gamma, random_state=seed)
    model = make_pipeline(StandardScaler(), CalibratedClassifierCV(svm, ensemble=False))
    model.fit(features, labels)
    return model


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
        return float(self.model.predict_proba(dmad_feature(reference, probe)[None, :])[0, 1])

    def score(self, sample: Sample) -> float:
        if sample.template is None or sample.probe_embedding is None:
            return 0.0
        return self.score_pair(sample.template, sample.probe_embedding)


def build(config: dict[str, Any]) -> Detector:
    return load_or_stub("m2_morph", [config["detectors"]["m2_morph"]["model"]], lambda ps: DMADMorph(ps[0]), config)
