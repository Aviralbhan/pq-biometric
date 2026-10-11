"""Shared plumbing for the evaluation scripts: connect to a server, enrol a user, get reason codes.

Two ways to reach the server:
- in-process (default): the FastAPI app runs inside the script; good for protocol tests.
- --url http://host:port: a real uvicorn server; reason codes and timings are read
  from its SQLite database (server.db_path in the same config), so run on the same machine.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from client.capture import load_session
from client.protocol import ApiClient
from preprocessing.video import encode_frames, encode_jpeg, unpack_jpegs
from settings import ROOT, resolve

RESULTS = ROOT / "eval" / "results"


@dataclass
class Connection:
    api: ApiClient
    db_path: Path
    server: Any = None  # the in-process Server, if any

    def request(self, request_id: int) -> dict[str, Any]:
        if self.server is not None:
            return self.server.db.get_request(request_id)
        with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
            cur = conn.execute(
                "SELECT request_id, challenge_id, mode, decision, reason_code, bytes_in, timing_json FROM requests WHERE request_id = ?",
                (request_id,),
            )
            row = cur.fetchone()
            out = dict(zip([d[0] for d in cur.description], row))
        out["timing"] = json.loads(out.pop("timing_json"))
        return out

    def reason(self, request_id: int) -> str | None:
        return self.request(request_id)["reason_code"]

    def verdict(self, challenge_id: str) -> dict[str, Any] | None:
        """The signed verdict record for a challenge (scores, versions, decision)."""
        if self.server is not None:
            row = self.server.db.get_verdict(challenge_id)
        else:
            with sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True) as conn:
                row = conn.execute("SELECT mode, record, signature FROM verdicts WHERE challenge_id = ?", (challenge_id,)).fetchone()
        return json.loads(row[1]) if row else None


@contextmanager
def connect(config: dict[str, Any], url: str | None = None) -> Iterator[Connection]:
    db_path = resolve(config["server"]["db_path"])
    if url:
        yield Connection(ApiClient(url), db_path)
        return
    from fastapi.testclient import TestClient

    from server.app import create_app

    app = create_app(config)
    with TestClient(app) as http:
        yield Connection(ApiClient(http), db_path, app.state.server)


def synthetic_session(n_frames: int = 30, fps: float = 30.0, seed: int = 0) -> tuple[bytes, dict[str, Any], bytes]:
    """A stand-in video when no recorded session is given. Real detectors will not find a face in it."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:480, 0:640]
    base = np.stack([xx % 256, yy % 256, (xx + yy) % 256], axis=-1).astype(np.uint8)
    frames = [np.clip(base.astype(int) + rng.integers(-8, 8, base.shape), 0, 255).astype(np.uint8) for _ in range(n_frames)]
    step = int(1e9 / fps)
    meta = {
        "device_name": "synthetic",
        "backend": "synthetic",
        "requested": {"width": 640, "height": 480, "fps": fps},
        "delivered": {"width": 640, "height": 480, "fps": fps},
        "frame_timestamps_ns": [i * step for i in range(n_frames)],
        "resolution_change_latency_ms": 80.0,
    }
    return encode_frames(frames, 85), meta, encode_jpeg(frames[0], 95)


def session_or_synthetic(session_dir: str | None, reference: str | None = None) -> tuple[bytes, dict[str, Any], bytes, str]:
    """(video, metadata, reference JPEG, description)."""
    if session_dir:
        video, meta = load_session(session_dir)
        ref = Path(reference).read_bytes() if reference else unpack_jpegs(video)[len(unpack_jpegs(video)) // 2]
        return video, meta, ref, f"session:{Path(session_dir).name}"
    video, meta, ref = synthetic_session()
    return video, meta, (Path(reference).read_bytes() if reference else ref), "synthetic"


def rel_path(path: str | Path) -> str:
    """Repo-relative POSIX path for result files (no local user directories in committed CSVs)."""
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_csv(path: str | Path, rows: Sequence[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for r in rows:
        fields += [k for k in r if k not in fields]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    return path


def percentile(values: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if len(values) else float("nan")
