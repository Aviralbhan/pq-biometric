"""Build the CelebA-Spoof CSVs for M1: keep faces >= 112 px wide, split by identity.

    python scripts/filter_celeba_spoof.py [--root data/celeba_spoof/CelebA_Spoof] [--val-fraction 0.1]

Label layout (CelebA-Spoof README, "Annotation in Json File"): [0:40] face attributes,
[40] spoof type, [41] illumination, [42] environment, [43] live/spoof. Spoof type
names are from the README's annotation table. Before trusting them the script
checks them against the dataset's own folder structure (every .../live/ image must
have live/spoof = 0 and spoof type 0; every .../spoof/ image 1 and non-zero) and
stops if they disagree.

The official split is train/test; val is carved from train identities.
Writes data/splits/celeba_spoof_{train,val,test}.csv with columns
    path, label (bona_fide|attack), attack_type, identity, split, face_w
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import ROOT  # noqa: E402

SPOOF_TYPE_INDEX, LIVE_SPOOF_INDEX = 40, 43
SPOOF_TYPES = {0: "live", 1: "photo", 2: "poster", 3: "a4", 4: "face_mask", 5: "upper_body_mask",
               6: "region_mask", 7: "pc", 8: "pad", 9: "phone", 10: "3d_mask"}
SCREEN_REPLAY = {"pc", "pad", "phone"}  # the screen-replay types reported separately for M4


def check_label_layout(labels: dict[str, list[int]]) -> None:
    bad = []
    for path, lab in labels.items():
        parts = Path(path).parts
        is_live_dir = "live" in parts
        if is_live_dir != (lab[LIVE_SPOOF_INDEX] == 0) or is_live_dir != (lab[SPOOF_TYPE_INDEX] == 0):
            bad.append(path)
        if lab[SPOOF_TYPE_INDEX] not in SPOOF_TYPES:
            bad.append(path)
    if bad:
        raise SystemExit(f"label layout does not match the folders for {len(bad)} images, e.g. {bad[:3]}; check the README")


def face_width(image_path: Path) -> int | None:
    """Face width in pixels from <image>_BB.txt (x y w h score on a 224x224 scale)."""
    bb = image_path.with_name(image_path.stem + "_BB.txt")
    if not bb.exists():
        return None
    try:
        _, _, w, _, _ = bb.read_text().split()[:5]
    except ValueError:
        return None
    img = cv2.imread(str(image_path))
    if img is None:
        return None
    return int(float(w) * img.shape[1] / 224)


def identity_of(rel_path: str) -> str:
    # Data/<split>/<identity>/<live|spoof>/<file>
    parts = Path(rel_path).parts
    return parts[parts.index("Data") + 2]


def build(root: Path, min_face: int, val_fraction: float, seed: int, limit: int | None) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for official in ("train", "test"):
        labels = json.loads((root / "metas" / "intra_test" / f"{official}_label.json").read_text(encoding="utf-8"))
        check_label_layout(labels)
        items = sorted(labels.items())[:limit]
        identities = sorted({identity_of(p) for p, _ in items})
        val_ids: set[str] = set()
        if official == "train":
            rng = random.Random(seed)
            val_ids = set(rng.sample(identities, max(1, int(len(identities) * val_fraction))))
        for rel, lab in items:
            path = root / rel
            w = face_width(path)
            if w is None or w < min_face:
                continue
            ident = identity_of(rel)
            split = "test" if official == "test" else ("val" if ident in val_ids else "train")
            kind = SPOOF_TYPES[lab[SPOOF_TYPE_INDEX]]
            out[split].append({"path": str(path), "label": "bona_fide" if kind == "live" else "attack",
                               "attack_type": "" if kind == "live" else kind, "identity": ident, "split": split, "face_w": w})
    ids = {k: {r["identity"] for r in v} for k, v in out.items()}
    overlap = (ids["train"] & ids["val"]) | (ids["train"] & ids["test"]) | (ids["val"] & ids["test"])
    if overlap:
        raise SystemExit(f"identity overlap between splits: {sorted(overlap)[:5]}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=str(ROOT / "data" / "celeba_spoof" / "CelebA_Spoof"))
    ap.add_argument("--min-face", type=int, default=112)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, help="max images per official split (quick runs)")
    ap.add_argument("--out-dir", default=str(ROOT / "data" / "splits"))
    args = ap.parse_args(argv)
    splits = build(Path(args.root), args.min_face, args.val_fraction, args.seed, args.limit)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"celeba_spoof_{name}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["path", "label", "attack_type", "identity", "split", "face_w"])
            w.writeheader()
            w.writerows(rows)
        counts = Counter(r["attack_type"] or "live" for r in rows)
        print(f"{name}: {len(rows)} images, {len({r['identity'] for r in rows})} identities -> {path}")
        print("   " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
