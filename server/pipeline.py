"""Detection stage: decode the video once, detect faces once, then run the matcher and M1-M5 in parallel.

ONNX Runtime and MediaPipe release the GIL while running, so a thread pool gives
real parallelism. Each module's own time is recorded as det_<name>.
"""

from __future__ import annotations

import asyncio
import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np

from crypto import Evidence
from detectors.base import Detector, Sample
from preprocessing.faces import FacePreprocessor
from preprocessing.video import decode_jpeg, sample_indices, unpack_jpegs
from server.timing import Timer

ATTACK_MODULES = ("m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection")


class Pipeline:
    def __init__(
        self,
        preprocessor: FacePreprocessor,
        detectors: dict[str, Any],
        sample_frames: int,
        executor: ThreadPoolExecutor,
    ):
        self.pre = preprocessor
        self.matcher = detectors["matcher"]
        self.modules: dict[str, Detector] = {k: detectors[k] for k in ATTACK_MODULES}
        self.sample_frames = sample_frames
        self.executor = executor

    def versions(self) -> dict[str, str]:
        out = {"face_detector": self.pre.version, "matcher": self.matcher.version}
        out.update({k: m.version for k, m in self.modules.items()})
        return out

    def prepare(self, evidence: Evidence, actions: list[str], template: np.ndarray | None, timer: Timer) -> Sample:
        with timer.stage("decode"):
            jpegs = unpack_jpegs(evidence.video)
            metadata = json.loads(evidence.metadata.decode("utf-8")) if evidence.metadata else {}
            frames = [decode_jpeg(jpegs[i]) for i in sample_indices(len(jpegs), self.sample_frames)]
        sample = Sample(jpegs=jpegs, metadata=metadata, actions=actions, frames=frames, template=template)
        with timer.stage("faces"):
            for frame in frames:
                crops = self.pre.process(frame)
                if crops is not None:
                    sample.crops.append(crops)
            if sample.crops:
                sample.best = max(sample.crops, key=lambda c: c.face.score)
        with timer.stage("embed"):
            if sample.best is not None:
                sample.probe_embedding = self.matcher.embed(sample.best.arcface112)
        return sample

    def _timed(self, name: str, fn, sample: Sample) -> tuple[str, float, float]:
        start = time.perf_counter_ns()
        value = float(fn(sample))
        return name, value, (time.perf_counter_ns() - start) / 1000.0

    async def analyse(
        self, evidence: Evidence, actions: list[str], template: np.ndarray | None, timer: Timer
    ) -> dict[str, float]:
        loop = asyncio.get_running_loop()
        sample = await loop.run_in_executor(self.executor, self.prepare, evidence, actions, template, timer)
        jobs = [("match", self.matcher.score)] + [(k, m.score) for k, m in self.modules.items()]
        start = time.perf_counter_ns()
        results = await asyncio.gather(*(loop.run_in_executor(self.executor, self._timed, n, f, sample) for n, f in jobs))
        timer.add("detectors_parallel", (time.perf_counter_ns() - start) / 1000.0)
        scores: dict[str, float] = {}
        for name, value, us in results:
            scores[name] = value
            timer.add(f"det_{name}", us)
        return scores
