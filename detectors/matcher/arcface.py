"""Face matcher: InsightFace buffalo_l ArcFace (w600k_r50.onnx), cosine similarity vs the enrolled template."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from detectors.base import STUB_VERSION, Sample, onnx_session
from settings import resolve

log = logging.getLogger(__name__)

EMBED_DIM = 512


def preprocess(crops_bgr: list[np.ndarray]) -> np.ndarray:
    """112x112 BGR crops -> NCHW float32 RGB in [-1, 1] (InsightFace convention)."""
    batch = np.stack([c[:, :, ::-1] for c in crops_bgr]).astype(np.float32)
    return ((batch - 127.5) / 127.5).transpose(0, 3, 1, 2).copy()


def l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(l2_normalize(a), l2_normalize(b)))


class ArcFaceMatcher:
    name = "matcher"

    def __init__(self, model_path: str | Path, version: str, threads: int = 2):
        self.version = version
        self._sess = onnx_session(model_path, threads)
        self._input = self._sess.get_inputs()[0].name

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        out = self._sess.run(None, {self._input: preprocess(crops_bgr)})[0]
        return l2_normalize(out.astype(np.float32))

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        return self.embed_batch([crop_bgr])[0]

    def score(self, sample: Sample) -> float:
        if sample.template is None or sample.probe_embedding is None:
            return -1.0
        return cosine(sample.template, sample.probe_embedding)


class MatcherStub:
    """Every face embeds to the same unit vector, so similarity is always 1.0."""

    name = "matcher"
    version = STUB_VERSION

    def embed(self, crop_bgr: np.ndarray) -> np.ndarray:
        v = np.zeros(EMBED_DIM, np.float32)
        v[0] = 1.0
        return v

    def embed_batch(self, crops_bgr: list[np.ndarray]) -> np.ndarray:
        return np.stack([self.embed(c) for c in crops_bgr]) if crops_bgr else np.zeros((0, EMBED_DIM), np.float32)

    def score(self, sample: Sample) -> float:
        if sample.template is None or sample.probe_embedding is None:
            return -1.0
        return cosine(sample.template, sample.probe_embedding)


def build(config: dict[str, Any]) -> ArcFaceMatcher | MatcherStub:
    c = config["detectors"]["matcher"]
    path = resolve(c["model"])
    if path.exists():
        return ArcFaceMatcher(path, c["version"], config["detectors"]["onnx_threads"])
    if config["detectors"].get("allow_stubs"):
        log.warning("matcher: weights missing (%s); using stub", path)
        return MatcherStub()
    raise FileNotFoundError(path)
