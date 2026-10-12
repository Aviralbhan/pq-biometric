from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from preprocessing.video import encode_frames
from settings import load_config

TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}


def stub_config(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Config that never touches real weights, Redis or var/: everything stubbed, in-process, under tmp_path."""
    missing = str(tmp_path / "missing.onnx")
    base = {
        "server": {"keys_dir": str(tmp_path / "keys"), "db_path": str(tmp_path / "db.sqlite"), "log_path": str(tmp_path / "server.log")},
        "redis": {"url": "fakeredis://"},
        "capture": {"device_dir": str(tmp_path / "device")},
        "preprocessing": {"yunet_model": missing},
        "detectors": {
            "allow_stubs": True,
            "matcher": {"model": missing},
            "m1_pad": {"models": [missing, missing]},
            "m2_morph": {"model": missing},
            "m3_deepfake": {"model": missing},
            "m4_replay": {"landmarker": missing},
            "m5_injection": {"model": missing},
        },
        # Fixed test thresholds: tests must not depend on whatever calibration config.yaml holds.
        "thresholds": {"version": "test", "match": 0.35, "m1_pad": 0.5, "m2_morph": 0.5, "m3_deepfake": 0.5, "m4_replay": 0.5, "m5_injection": 0.5},
    }
    from settings import deep_merge

    return load_config(overrides=deep_merge(base, overrides))


def synthetic_frames(n: int = 8, h: int = 480, w: int = 640, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.integers(0, 255, (h, w, 3), dtype=np.uint8) for _ in range(n)]


def synthetic_video(n: int = 8, seed: int = 0) -> bytes:
    return encode_frames(synthetic_frames(n, seed=seed), quality=70)


def synthetic_metadata(n: int = 8, fps: float = 30.0, device: str = "Integrated Webcam") -> dict[str, Any]:
    step = int(1e9 / fps)
    return {
        "device_name": device,
        "backend": "test",
        "requested": {"width": 640, "height": 480, "fps": 30},
        "delivered": {"width": 640, "height": 480, "fps": fps},
        "frame_timestamps_ns": [i * step + (i % 3) * 1_700_000 for i in range(n)],
        "resolution_change_latency_ms": 85.0,
    }


@pytest.fixture
def config(tmp_path):
    return stub_config(tmp_path)
