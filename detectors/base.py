"""Shared detector interface.

Every module scores one Sample. Attack modules (M1-M5) return a score in [0, 1]
where higher means "more likely an attack"; the decision rule rejects when the
score is above that module's threshold. The matcher returns cosine similarity.

When a module's weights are missing and `detectors.allow_stubs` is true, a Stub
with a fixed score and version "stub" is used instead. The version goes into the
signed verdict, so a stubbed run can never be mistaken for a real one.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from preprocessing.faces import FaceCrops
from settings import resolve

log = logging.getLogger(__name__)

STUB_VERSION = "stub"


@dataclass
class Sample:
    """Everything the detectors may look at for one verification request."""

    jpegs: list[bytes]  # every captured frame, JPEG-encoded (M4, M5)
    metadata: dict[str, Any]  # capture metadata reported by the client (M5)
    actions: list[str]  # liveness actions issued with the challenge (M4)
    frames: list[np.ndarray] = field(default_factory=list)  # sampled decoded frames
    crops: list[FaceCrops] = field(default_factory=list)  # faces found in the sampled frames
    best: FaceCrops | None = None  # highest-scoring face, used by the matcher and M2
    template: np.ndarray | None = None  # enrolled ArcFace embedding
    probe_embedding: np.ndarray | None = None  # ArcFace embedding of `best`


class Detector(ABC):
    name: str
    version: str

    @abstractmethod
    def score(self, sample: Sample) -> float: ...


class Stub(Detector):
    def __init__(self, name: str, value: float = 0.0):
        self.name = name
        self.version = STUB_VERSION
        self.value = value

    def score(self, sample: Sample) -> float:
        return self.value


def onnx_session(path: str | Path, threads: int = 2):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=opts, providers=["CPUExecutionProvider"])


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - x.max(axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)


def load_or_stub(
    name: str,
    paths: list[str],
    factory: Callable[[list[Path]], Detector],
    config: dict[str, Any],
    stub_value: float = 0.0,
) -> Detector:
    """Build a detector from weight files, or a Stub if any is missing and stubs are allowed."""
    resolved = [resolve(p) for p in paths]
    missing = [p for p in resolved if not p.exists()]
    if not missing:
        return factory(resolved)
    if config["detectors"].get("allow_stubs"):
        log.warning("%s: weights missing (%s); using stub", name, ", ".join(map(str, missing)))
        return Stub(name, stub_value)
    raise FileNotFoundError(f"{name}: missing {missing}")
