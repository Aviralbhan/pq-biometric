"""M5 injection features from capture metadata and frame timing.

Virtual cameras (OBS Virtual Camera, v4l2loopback) tend to show: a known device
name, delivered resolution/FPS that ignore what was requested, unnaturally regular
or bursty frame intervals, repeated identical frames, and near-zero latency when
the resolution is changed.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

FEATURE_NAMES = (
    "virtual_name",
    "name_missing",
    "res_mismatch",
    "fps_ratio",
    "measured_fps",
    "n_frames",
    "dt_mean_ms",
    "dt_std_ms",
    "dt_cv",
    "dt_min_ms",
    "dt_max_ms",
    "dt_p05_ms",
    "dt_p95_ms",
    "dt_near_median",
    "dup_frame_ratio",
    "res_change_latency_ms",
)


def is_virtual_name(name: str | None, virtual_names: list[str]) -> bool:
    lowered = (name or "").lower()
    return any(v.lower() in lowered for v in virtual_names)


def extract(metadata: dict[str, Any], jpegs: list[bytes], virtual_names: list[str]) -> dict[str, float]:
    name = metadata.get("device_name")
    req = metadata.get("requested", {}) or {}
    got = metadata.get("delivered", {}) or {}
    ts = np.asarray(metadata.get("frame_timestamps_ns") or [], dtype=np.float64)
    dt = np.diff(ts) / 1e6 if ts.size >= 2 else np.zeros(1)
    med = float(np.median(dt))

    hashes = [hashlib.blake2b(j, digest_size=16).digest() for j in jpegs]
    dups = sum(a == b for a, b in zip(hashes, hashes[1:]))
    span_s = (ts[-1] - ts[0]) / 1e9 if ts.size >= 2 else 0.0
    latency = metadata.get("resolution_change_latency_ms")

    return {
        "virtual_name": float(is_virtual_name(name, virtual_names)),
        "name_missing": float(not name or str(name).lower() == "unknown"),
        "res_mismatch": float((req.get("width"), req.get("height")) != (got.get("width"), got.get("height"))),
        "fps_ratio": float(got.get("fps") or 0.0) / float(req.get("fps") or 1.0),
        "measured_fps": (ts.size - 1) / span_s if span_s > 0 else 0.0,
        "n_frames": float(len(jpegs)),
        "dt_mean_ms": float(dt.mean()),
        "dt_std_ms": float(dt.std()),
        "dt_cv": float(dt.std() / dt.mean()) if dt.mean() > 0 else 0.0,
        "dt_min_ms": float(dt.min()),
        "dt_max_ms": float(dt.max()),
        "dt_p05_ms": float(np.percentile(dt, 5)),
        "dt_p95_ms": float(np.percentile(dt, 95)),
        "dt_near_median": float(np.mean(np.abs(dt - med) < 1.0)),
        "dup_frame_ratio": dups / max(len(jpegs) - 1, 1),
        "res_change_latency_ms": float(latency) if latency is not None else -1.0,
    }


def to_vector(features: dict[str, float]) -> np.ndarray:
    return np.array([features[k] for k in FEATURE_NAMES], dtype=np.float64)
