"""M5 injection detector: rules first, then a HistGradientBoosting classifier.

Rule: a device name matching a known virtual camera scores 1.0 outright.
Otherwise the classifier's P(injected) is the score. Without a trained model
(models/weights/m5_injection/hgb.joblib) only the rule applies; the module then
reports version "<version>-rules-only" so the verdict shows it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from detectors.base import Detector, Sample
from detectors.m5_injection.features import FEATURE_NAMES, extract, to_vector
from settings import resolve


def train(x: np.ndarray, y: np.ndarray, seed: int = 0) -> HistGradientBoostingClassifier:
    # min_samples_leaf=5: M5 sets are small (~50 sessions per class); the default of 20 blocks most splits.
    model = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, min_samples_leaf=5, random_state=seed)
    model.fit(x, y)
    return model


def save(model: HistGradientBoostingClassifier, path: str | Path, version: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "version": version, "features": list(FEATURE_NAMES)}, path)


class InjectionDetector(Detector):
    name = "m5_injection"

    def __init__(self, virtual_names: list[str], version: str, model_path: Path | None = None):
        self.virtual_names = virtual_names
        self.model = None
        self.version = f"{version}-rules-only"
        if model_path is not None and model_path.exists():
            bundle = joblib.load(model_path)
            if list(bundle["features"]) != list(FEATURE_NAMES):
                raise ValueError("M5 model was trained on a different feature set")
            self.model = bundle["model"]
            self.version = bundle["version"]

    def score_features(self, features: dict[str, float]) -> float:
        if features["virtual_name"]:
            return 1.0
        if self.model is None:
            return 0.0
        return float(self.model.predict_proba(to_vector(features)[None, :])[0, 1])

    def score(self, sample: Sample) -> float:
        return self.score_features(extract(sample.metadata, sample.jpegs, self.virtual_names))


def build(config: dict[str, Any]) -> Detector:
    c = config["detectors"]["m5_injection"]
    return InjectionDetector(c["virtual_camera_names"], c["version"], resolve(c["model"]))
