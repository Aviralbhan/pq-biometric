"""Capture client (Layer A): challenge -> actions -> 4 s video -> metadata -> manifest -> sign -> seal -> verify.

    python -m client.capture --user alice --tx '{"type":"payment","amount":500,"currency":"INR","payee":"acct-42"}'

Options:
    --mode PQ|CLASSICAL       crypto mode (default: crypto.mode in config.yaml)
    --save DIR                also save the recorded session (video.beav + metadata.json)
    --from-session DIR        send a previously saved session instead of recording (pre-recorded video attack)
    --camera N                camera index (default: capture.camera_index)
    --no-window               record without a preview window
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from client.protocol import ApiClient, DeviceIdentity, ProtocolError, check_challenge, seal_evidence
from crypto import Mode
from preprocessing.video import encode_jpeg, pack_jpegs
from settings import load_config, resolve

ACTION_TEXT = {"blink_twice": "Blink twice", "turn_left": "Turn your head LEFT", "turn_right": "Turn your head RIGHT"}
WINDOW = "PQ biometric capture"


class FrameSource(Protocol):
    def read(self) -> tuple[bool, np.ndarray | None]: ...
    def set(self, prop: int, value: float) -> bool: ...
    def get(self, prop: int) -> float: ...
    def release(self) -> None: ...


def camera_api() -> int:
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    if system == "Linux":
        return cv2.CAP_V4L2
    return cv2.CAP_ANY


def device_name(index: int) -> str | None:
    """Camera name as the OS reports it (M5 checks it against known virtual cameras)."""
    system = platform.system()
    try:
        if system == "Linux":
            return Path(f"/sys/class/video4linux/video{index}/name").read_text().strip()
        if system == "Windows":
            from pygrabber.dshow_graph import FilterGraph

            names = FilterGraph().get_input_devices()
            return names[index] if index < len(names) else None
    except Exception:
        return None
    return None


def open_camera(index: int, width: int, height: int, fps: float) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, camera_api())
    if not cap.isOpened():
        raise RuntimeError(f"cannot open camera {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _wait_for_size(cap: FrameSource, size: tuple[int, int], timeout_s: float) -> float | None:
    start = time.perf_counter()
    while time.perf_counter() - start < timeout_s:
        ok, frame = cap.read()
        if ok and frame is not None and (frame.shape[1], frame.shape[0]) == size:
            return (time.perf_counter() - start) * 1000.0
    return None


def measure_resolution_change(cap: FrameSource, probe: tuple[int, int], target: tuple[int, int], timeout_s: float = 2.0) -> float | None:
    """Time (ms) until frames arrive at a newly requested resolution. Virtual cameras often never switch."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, probe[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, probe[1])
    latency = _wait_for_size(cap, probe, timeout_s)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, target[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target[1])
    _wait_for_size(cap, target, timeout_s)
    return latency


def draw_overlay(frame: np.ndarray, lines: list[str], progress: float | None, highlight: int | None) -> np.ndarray:
    view = cv2.flip(frame, 1)  # mirror the preview only; the recorded frame is untouched
    for i, text in enumerate(lines):
        color = (0, 255, 255) if i == highlight else (255, 255, 255)
        cv2.putText(view, text, (16, 36 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(view, text, (16, 36 + 32 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)
    if progress is not None:
        h, w = view.shape[:2]
        cv2.rectangle(view, (0, h - 12), (int(w * progress), h), (0, 200, 0), -1)
    return view


@dataclass
class Recording:
    jpegs: list[bytes]
    timestamps_ns: list[int]
    size: tuple[int, int]


def record(
    cap: FrameSource,
    duration_s: float,
    actions: list[str],
    quality: int,
    header: list[str] | None = None,
    show: bool = True,
    countdown_s: float = 3.0,
) -> Recording:
    header = header or []
    prompts = [f"{i + 1}. {ACTION_TEXT.get(a, a)}" for i, a in enumerate(actions)]
    if show:
        end = time.perf_counter() + countdown_s
        while (left := end - time.perf_counter()) > 0:
            ok, frame = cap.read()
            if ok:
                cv2.imshow(WINDOW, draw_overlay(frame, header + prompts + [f"Starting in {left:.0f}..."], None, None))
                cv2.waitKey(1)
    jpegs: list[bytes] = []
    stamps: list[int] = []
    size = (0, 0)
    start = time.perf_counter_ns()
    while (elapsed := time.perf_counter_ns() - start) < duration_s * 1e9:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        stamps.append(time.perf_counter_ns())
        size = (frame.shape[1], frame.shape[0])
        jpegs.append(encode_jpeg(frame, quality))
        if show:
            progress = elapsed / (duration_s * 1e9)
            current = min(int(progress * len(actions)), len(actions) - 1) if actions else None
            cv2.imshow(WINDOW, draw_overlay(frame, header + prompts, progress, None if current is None else len(header) + current))
            cv2.waitKey(1)
    if show:
        cv2.destroyWindow(WINDOW)
    if not jpegs:
        raise RuntimeError("camera delivered no frames")
    return Recording(jpegs, stamps, size)


def capture_session(
    config: dict[str, Any],
    actions: list[str],
    header: list[str] | None = None,
    camera_index: int | None = None,
    show: bool = True,
    cap: FrameSource | None = None,
    name: str | None = None,
) -> tuple[bytes, dict[str, Any]]:
    """Record one session. Returns (video bytes, metadata dict)."""
    c = config["capture"]
    index = c["camera_index"] if camera_index is None else camera_index
    w, h, fps = int(c["width"]), int(c["height"]), float(c["fps"])
    own = cap is None
    cap = cap or open_camera(index, w, h, fps)
    try:
        latency = measure_resolution_change(cap, tuple(c["resolution_probe"]), (w, h))
        delivered_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        rec = record(cap, float(c["duration_seconds"]), actions, int(c["jpeg_quality"]), header, show)
    finally:
        if own:
            cap.release()
    metadata = {
        "device_name": name if name is not None else device_name(index),
        "backend": f"opencv-{cv2.__version__}/{platform.system()}",
        "requested": {"width": w, "height": h, "fps": fps},
        "delivered": {"width": rec.size[0], "height": rec.size[1], "fps": delivered_fps},
        "frame_timestamps_ns": [t - rec.timestamps_ns[0] for t in rec.timestamps_ns],
        "resolution_change_latency_ms": latency,
    }
    return pack_jpegs(rec.jpegs), metadata


def save_session(directory: str | Path, video: bytes, metadata: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    (d / "video.beav").write_bytes(video)
    (d / "metadata.json").write_text(json.dumps({**metadata, **(extra or {})}, indent=1), encoding="utf-8")
    return d


def load_session(directory: str | Path) -> tuple[bytes, dict[str, Any]]:
    d = Path(directory)
    meta = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
    for key in ("label", "actions", "camera_tool", "recorded_at"):
        meta.pop(key, None)
    return (d / "video.beav").read_bytes(), meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True)
    ap.add_argument("--tx", required=True, help="transaction context as JSON")
    ap.add_argument("--mode", choices=[m.value for m in Mode])
    ap.add_argument("--camera", type=int)
    ap.add_argument("--save")
    ap.add_argument("--from-session")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    mode = Mode(args.mode or config["crypto"]["mode"])
    tx = json.loads(args.tx)
    identity = DeviceIdentity.load(resolve(config["capture"]["device_dir"]), args.user, mode)
    api = ApiClient(config["capture"]["server_url"])

    ch = api.challenge(args.user, tx, mode)
    try:
        check_challenge(identity, ch, tx)
    except ProtocolError as exc:
        print(f"Refusing challenge: {exc}", file=sys.stderr)
        return 2
    print(f"Transaction: {json.dumps(tx, sort_keys=True)}")
    print(f"Challenge {ch['challenge_id']} ({mode.value}); actions: {', '.join(ch['actions'])}")

    if args.from_session:
        video, metadata = load_session(args.from_session)
        print(f"Sending pre-recorded session from {args.from_session}")
    else:
        header = [f"{k}: {v}" for k, v in sorted(tx.items())][:3]
        video, metadata = capture_session(config, ch["actions"], header, args.camera, show=not args.no_window)
        if args.save:
            save_session(args.save, video, metadata, {"actions": ch["actions"]})

    t0 = time.perf_counter()
    resp, request_id = api.verify(seal_evidence(identity, ch, video, metadata))
    print(f"Decision: {resp['decision'].upper()}  (round trip {1000 * (time.perf_counter() - t0):.0f} ms, request {request_id})")
    return 0 if resp["decision"] == "accepted" else 1


if __name__ == "__main__":
    raise SystemExit(main())
