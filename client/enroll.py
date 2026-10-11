"""Enrolment: generate device signing keys, send the public key and a reference photo, pin the server keys.

    python -m client.enroll --user alice                  # take the reference photo with the webcam (SPACE)
    python -m client.enroll --user alice --photo me.jpg   # or use an existing photo
    python -m client.enroll --user alice --modes PQ       # default: every mode in crypto.allowed_modes
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

from client.capture import WINDOW, draw_overlay, open_camera
from client.protocol import ApiClient
from crypto import Mode
from preprocessing.video import encode_jpeg
from settings import load_config, resolve


def take_photo(config: dict, camera_index: int | None = None) -> bytes:
    c = config["capture"]
    cap = open_camera(c["camera_index"] if camera_index is None else camera_index, c["width"], c["height"], c["fps"])
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue
            cv2.imshow(WINDOW, draw_overlay(frame, ["Look at the camera", "SPACE = take photo, ESC = cancel"], None, None))
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                raise SystemExit("cancelled")
            if key == 32:
                return encode_jpeg(frame, 95)
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--user", required=True)
    ap.add_argument("--photo")
    ap.add_argument("--modes", nargs="+", choices=[m.value for m in Mode])
    ap.add_argument("--camera", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    config = load_config(args.config)
    modes = [Mode(m) for m in (args.modes or config["crypto"]["allowed_modes"])]
    photo = Path(args.photo).read_bytes() if args.photo else take_photo(config, args.camera)
    api = ApiClient(config["capture"]["server_url"])
    device_dir = resolve(config["capture"]["device_dir"])
    for mode in modes:
        try:
            identity = api.enroll(args.user, mode, photo)
        except Exception as exc:
            print(f"{mode.value}: enrolment failed: {exc}", file=sys.stderr)
            return 1
        path = identity.save(device_dir)
        print(f"{mode.value}: enrolled {args.user}; keys in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
