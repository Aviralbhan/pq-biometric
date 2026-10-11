"""Attack x mode matrix: which check or module stops each attack, with the detectors held constant.

    python -m eval.attack_matrix \
        --attack print=data/attack_sets/print --attack screen=data/attack_sets/screen \
        --attack deepfake=data/attack_sets/openforensics_fake --attack obs=var/sessions/obs \
        --bona-fide genuine=var/sessions/bona_fide --morph-pairs data/attack_sets/morph_pairs.csv \
        [--reference me.jpg] [--limit 200] [--modes CLASSICAL PQ]

Sources: a directory of recorded sessions (video.beav + metadata.json, from
client.capture --save or scripts/record_m5.py) or of images (.jpg/.png). An image is
sent as a static 4 s video with webcam-like metadata and is also used as the enrolled
reference, so the matcher passes and the attack detectors are what is measured.
Sessions enrol with --reference (the genuine user's photo) if given, else a frame of
the session. Morph pairs (CSV columns: morph, probe) enrol the morph and probe with
the bona fide image.

Writes eval/results/attack_matrix_samples.csv (every sample, all scores) and
eval/results/attack_matrix.csv (per attack and mode: rejection rate, reason codes,
how often each module alone would have flagged it). Protocol attacks from
eval/results/replay.csv are appended when that file exists.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from client.capture import load_session
from client.protocol import check_challenge, seal_evidence
from crypto import Mode
from eval.harness import RESULTS, connect, write_csv
from preprocessing.video import encode_frames, encode_jpeg, unpack_jpegs
from settings import load_config

TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
MODULES = ("m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection")


def letterbox(img: np.ndarray, w: int = 640, h: int = 480) -> np.ndarray:
    scale = min(w / img.shape[1], h / img.shape[0])
    resized = cv2.resize(img, (max(1, int(img.shape[1] * scale)), max(1, int(img.shape[0] * scale))))
    out = np.zeros((h, w, 3), np.uint8)
    y, x = (h - resized.shape[0]) // 2, (w - resized.shape[1]) // 2
    out[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return out


def static_video(img: np.ndarray, n_frames: int = 60, fps: float = 30.0, seed: int = 0) -> tuple[bytes, dict[str, Any]]:
    frame = letterbox(img)
    rng = np.random.default_rng(seed)
    step = 1e9 / fps
    stamps = np.cumsum(np.r_[0, rng.normal(step, step * 0.08, n_frames - 1)]).astype(np.int64)
    meta = {
        "device_name": "image-source",
        "backend": "attack_matrix",
        "requested": {"width": 640, "height": 480, "fps": fps},
        "delivered": {"width": 640, "height": 480, "fps": fps},
        "frame_timestamps_ns": [int(t) for t in stamps],
        "resolution_change_latency_ms": 90.0,
    }
    return encode_frames([frame] * n_frames, 90), meta


def iter_source(path: Path, limit: int | None):
    """Yield (sample_id, video, metadata, reference_jpeg_or_None)."""
    items = sorted(p for p in path.iterdir() if (p / "video.beav").exists() or p.suffix.lower() in IMAGE_EXT)
    for p in items[:limit]:
        if p.is_dir():
            video, meta = load_session(p)
            yield p.name, video, meta, None
        else:
            img = cv2.imread(str(p))
            if img is None:
                continue
            video, meta = static_video(img)
            yield p.name, video, meta, encode_jpeg(letterbox(img), 95)


def iter_morph_pairs(csv_path: Path, limit: int | None):
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:limit]
    for i, r in enumerate(rows):
        morph, probe = cv2.imread(r["morph"]), cv2.imread(r["probe"])
        if morph is None or probe is None:
            continue
        video, meta = static_video(probe, seed=i)
        yield f"{Path(r['morph']).name}|{Path(r['probe']).name}", video, meta, encode_jpeg(letterbox(morph), 95)


def run_sample(conn, mode: Mode, user: str, video: bytes, meta: dict, ref: bytes) -> dict[str, Any]:
    try:
        identity = conn.api.enroll(user, mode, ref)
    except Exception as exc:  # no face in the reference: the attack never gets an enrolment
        return {"enrolled": False, "decision": "not_enrolled", "reason_code": type(exc).__name__}
    ch = conn.api.challenge(user, TX, mode)
    check_challenge(identity, ch, TX)
    resp, rid = conn.api.verify(seal_evidence(identity, ch, video, meta))
    record = conn.verdict(ch["challenge_id"]) or {}
    row = {"enrolled": True, "decision": resp["decision"], "reason_code": record.get("reason_code")}
    row.update({f"score_{k}": v for k, v in (record.get("scores") or {}).items()})
    row["model_versions"] = json.dumps(record.get("model_versions"), sort_keys=True)
    return row


def summarise(samples: list[dict], thresholds: dict[str, Any]) -> list[dict]:
    groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for s in samples:
        groups[(s["mode"], s["kind"], s["attack"])].append(s)
    rows = []
    for (mode, kind, attack), items in sorted(groups.items()):
        enrolled = [s for s in items if s["enrolled"]]
        row: dict[str, Any] = {
            "mode": mode, "kind": kind, "attack": attack, "n": len(items), "n_enrolled": len(enrolled),
            "rejection_rate": round(sum(s["decision"] != "accepted" for s in items) / len(items), 4),
            "reason_codes": ";".join(f"{k}={v}" for k, v in Counter(str(s["reason_code"]) for s in items).most_common()),
        }
        for m in MODULES:
            vals = [s.get(f"score_{m}") for s in enrolled if s.get(f"score_{m}") is not None]
            row[f"flag_rate[{m}]"] = round(float(np.mean([v > float(thresholds[m]) for v in vals])), 4) if vals else None
        sims = [s.get("score_match") for s in enrolled if s.get("score_match") is not None]
        row["no_match_rate"] = round(float(np.mean([v < float(thresholds["match"]) for v in sims])), 4) if sims else None
        rows.append(row)
    return rows


def replay_rows() -> list[dict]:
    path = RESULTS / "replay.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        return [
            {
                "mode": r["mode"], "kind": "protocol", "attack": r["case"], "n": int(r["runs"]), "n_enrolled": int(r["runs"]),
                "rejection_rate": float(r["rejection_rate"]),
                "reason_codes": f"{r['expected_code'] or 'none'}={r['correct_code']}" + (f";other={r['other_codes']}" if r["other_codes"] else ""),
            }
            for r in csv.DictReader(fh)
        ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--attack", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--bona-fide", action="append", default=[], metavar="NAME=DIR")
    ap.add_argument("--morph-pairs")
    ap.add_argument("--reference")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--modes", nargs="+", default=["CLASSICAL", "PQ"], choices=[m.value for m in Mode])
    ap.add_argument("--url")
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    ref_override = Path(args.reference).read_bytes() if args.reference else None
    sources = [("attack", *a.split("=", 1)) for a in args.attack] + [("bona_fide", *b.split("=", 1)) for b in args.bona_fide]
    samples: list[dict] = []
    with connect(config, args.url) as conn:
        for mode in map(Mode, args.modes):
            for kind, name, path in sources:
                for i, (sid, video, meta, ref) in enumerate(iter_source(Path(path), args.limit)):
                    ref = ref_override or ref or unpack_jpegs(video)[len(unpack_jpegs(video)) // 2]
                    row = run_sample(conn, mode, f"am-{name}-{mode.value[0]}{i}", video, meta, ref)
                    samples.append({"mode": mode.value, "kind": kind, "attack": name, "sample": sid, **row})
                print(f"  {mode.value} {name}: {sum(s['attack'] == name and s['mode'] == mode.value for s in samples)} samples")
            if args.morph_pairs:
                for i, (sid, video, meta, ref) in enumerate(iter_morph_pairs(Path(args.morph_pairs), args.limit)):
                    row = run_sample(conn, mode, f"am-morph-{mode.value[0]}{i}", video, meta, ref)
                    samples.append({"mode": mode.value, "kind": "attack", "attack": "morph_enrolment", "sample": sid, **row})
    if not samples:
        ap.error("no samples: give --attack, --bona-fide or --morph-pairs")
    print(f"wrote {write_csv(RESULTS / 'attack_matrix_samples.csv', samples)}")
    rows = summarise(samples, config["thresholds"]) + replay_rows()
    print(f"wrote {write_csv(RESULTS / 'attack_matrix.csv', rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
