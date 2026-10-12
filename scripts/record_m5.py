"""Record M5 training sessions: real webcam vs virtual cameras (OBS Virtual Camera, v4l2loopback).

    python scripts/record_m5.py --label bona_fide    --camera 0 --subject aviral --sessions 50
    python scripts/record_m5.py --label obs --source obs_video --camera 1 --subject aviral --sessions 25   # OBS Virtual Camera playing a video
    python scripts/record_m5.py --label obs --source obs_image --camera 1 --subject aviral --sessions 25   # OBS Virtual Camera showing a still image
    python scripts/record_m5.py --label v4l2loopback --camera 4 --subject aviral --sessions 50   # Linux: ffmpeg -> /dev/video4

Sessions go to var/m5_sessions/<label>/<subject>_<n>/ (video.beav + metadata.json),
recorded exactly as the capture client records them, with random actions shown so
behaviour looks like a real transaction.

Consent: only team members, each with written consent. The subject must be listed in
docs/consent.txt (one name per line) or recording refuses to start.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.capture import capture_session, device_name, save_session  # noqa: E402
from settings import ROOT, load_config  # noqa: E402

LABELS = ("bona_fide", "obs", "v4l2loopback")
OUT = ROOT / "var" / "m5_sessions"
CONSENT = ROOT / "docs" / "consent.txt"


def consented(subject: str) -> bool:
    if not CONSENT.exists():
        return False
    names = {line.strip().lower() for line in CONSENT.read_text(encoding="utf-8").splitlines() if line.strip() and not line.startswith("#")}
    return subject.lower() in names


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, choices=LABELS)
    ap.add_argument("--camera", type=int, required=True)
    ap.add_argument("--subject", required=True)
    ap.add_argument("--source", help="what the camera showed, e.g. obs_video / obs_image (default: the label)")
    ap.add_argument("--sessions", type=int, default=50)
    ap.add_argument("--pause", type=float, default=1.0, help="seconds between sessions")
    ap.add_argument("--no-window", action="store_true")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    if not consented(args.subject):
        print(f"{args.subject} is not listed in {CONSENT}; get written consent first.", file=sys.stderr)
        return 2
    config = load_config(args.config)
    pool = config["challenge"]["actions"]
    name = device_name(args.camera)
    print(f"camera {args.camera}: {name!r} -> label {args.label}")
    out_dir = OUT / args.label
    prefix = f"{args.subject}_{args.source}" if args.source else args.subject
    start = len(list(out_dir.glob(f"{prefix}_*"))) if out_dir.exists() else 0
    for i in range(start, start + args.sessions):
        actions = random.sample(pool, config["challenge"]["num_actions"])
        video, meta = capture_session(config, actions, [f"{args.label} session {i + 1 - start}/{args.sessions}"], args.camera, show=not args.no_window, name=name)
        d = save_session(out_dir / f"{prefix}_{i:03d}", video, meta,
                         {"label": args.label, "actions": actions, "camera_tool": args.source or args.label, "recorded_at": int(time.time())})
        print(f"  saved {d} ({len(meta['frame_timestamps_ns'])} frames)")
        time.sleep(args.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
