"""Download OpenForensics Val and/or Test-Dev (CC BY 4.0, Zenodo 5528418) and crop every annotated face.

    python scripts/download_openforensics.py --splits Val            # 3.1 GB + 122 MB annotations
    python scripts/download_openforensics.py --splits Val Test-Dev   # + 8.2 GB + 313 MB

OpenForensics images are multi-face scenes with COCO-style annotations: every face
has a bbox and a category (real or fake; names are read from the annotation file).
Faces narrower than --min-face px are skipped. Crops (box enlarged by 1.3, square)
go to data/openforensics/crops/<split>/<category>/, and data/splits/m3_<split>.csv
lists them with columns path, label (bona_fide|attack), attack_type, split, source_image.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from preprocessing.faces import crop_square  # noqa: E402
from scripts._download import download, extract, remove  # noqa: E402
from settings import ROOT  # noqa: E402

RECORD = "https://zenodo.org/api/records/5528418"
DEST = ROOT / "data" / "openforensics"
SPLITS = ROOT / "data" / "splits"


def crop_split(split: str, images_root: Path, annotations: Path, out_root: Path, min_face: int, limit: int | None) -> list[dict]:
    coco = json.loads(annotations.read_text(encoding="utf-8"))
    categories = {c["id"]: c["name"].lower() for c in coco["categories"]}
    if not {"real", "fake"} <= set(categories.values()):
        raise SystemExit(f"unexpected categories in {annotations.name}: {categories}")
    images = {im["id"]: im["file_name"] for im in coco["images"]}
    index = {p.name: p for p in images_root.rglob("*") if p.suffix.lower() in {".jpg", ".png"}}
    rows, cache = [], {}
    for ann in coco["annotations"][:limit]:
        name = Path(images[ann["image_id"]]).name
        if name not in index:
            continue
        x, y, w, h = ann["bbox"]
        if min(w, h) < min_face:
            continue
        if name not in cache:
            cache.clear()
            cache[name] = cv2.imread(str(index[name]))
        img = cache[name]
        if img is None:
            continue
        cat = categories[ann["category_id"]]
        out = out_root / split / cat / f"{Path(name).stem}_{ann['id']}.jpg"
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), crop_square(img, (x, y, w, h), 1.3, 224), [cv2.IMWRITE_JPEG_QUALITY, 95])
        rows.append({"path": str(out), "label": "bona_fide" if cat == "real" else "attack",
                     "attack_type": "" if cat == "real" else "openforensics_fake", "split": split.lower(), "source_image": name})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", nargs="+", default=["Val"], choices=["Val", "Test-Dev"])
    ap.add_argument("--min-face", type=int, default=112)
    ap.add_argument("--limit", type=int, help="max annotations per split (quick runs)")
    ap.add_argument("--keep-archives", action="store_true")
    args = ap.parse_args(argv)
    with urllib.request.urlopen(RECORD, timeout=60) as resp:
        files = {f["key"]: f for f in json.load(resp)["files"]}
    SPLITS.mkdir(parents=True, exist_ok=True)
    for split in args.splits:
        parts = sorted(k for k in files if k.startswith(f"{split}.zip") or k.startswith(f"{split}_part_"))
        ann = download(files[f"{split}_poly.json"]["links"]["self"], DEST / f"{split}_poly.json", files[f"{split}_poly.json"]["size"])
        for key in parts:
            archive = download(files[key]["links"]["self"], DEST / "archives" / key, files[key]["size"])
            extract(archive, DEST / "images" / split)
            if not args.keep_archives:
                remove(archive)
        rows = crop_split(split, DEST / "images" / split, ann, DEST / "crops", args.min_face, args.limit)
        out = SPLITS / f"m3_{split.lower()}.csv"
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"{split}: {sum(r['label'] == 'bona_fide' for r in rows)} real, {sum(r['label'] == 'attack' for r in rows)} fake faces -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
