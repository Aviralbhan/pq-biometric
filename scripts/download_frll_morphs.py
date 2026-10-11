"""Download FRLL-Morphs (CC BY-SA 4.0, Zenodo 4415159, ~212 MB) into data/frll_morphs/ and build the M2 pair list.

    python scripts/download_frll.py          # first: the bona fide FRLL images
    python scripts/download_frll_morphs.py

Writes data/splits/m2_pairs.csv with columns
    reference, probe, label (morph|bona_fide), morph_method, subject_a, subject_b, reference_view, probe_view
Pairs (THREE_QUARTER = the four 3/4 views):
    bona fide, both directions:
        reference = neutral_front of A, probe = smiling_front or a 3/4 view of A
        reference = smiling_front of A, probe = neutral_front or a 3/4 view of A
    morph: reference = morph of A and B, probe = smiling_front or a 3/4 view of A, and of B
           (not neutral_front: the morphs are made from the neutral front images)
Morph probes use the same 3/4 views as bona fide probes, so head pose cannot separate the classes.
No split column: scripts/train_m2.py assigns 5 identity-grouped folds (GroupKFold).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._download import download, extract  # noqa: E402
from settings import ROOT  # noqa: E402

RECORD = "https://zenodo.org/api/records/4415159"
DEST = ROOT / "data" / "frll_morphs"
FRLL = ROOT / "data" / "frll"
SPLITS = ROOT / "data" / "splits"
IMAGE_EXT = {".jpg", ".jpeg", ".png"}
PAIR_RE = re.compile(r"(\d{3})\D+(\d{3})")
SUBJECT_RE = re.compile(r"^(\d{3})")


def subject_images(folder: Path) -> dict[str, Path]:
    """FRLL file names start with the 3-digit subject id (e.g. 001_03.jpg)."""
    out = {}
    for p in sorted(folder.rglob("*")):
        m = SUBJECT_RE.match(p.name)
        if p.suffix.lower() in IMAGE_EXT and m and "__MACOSX" not in p.parts:
            out.setdefault(m.group(1), p)
    return out


def find_morphs(root: Path) -> list[tuple[Path, str, str, str]]:
    """(path, method, subject_a, subject_b). The method is the image's folder name without "morph_"."""
    morphs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        m = PAIR_RE.search(p.stem)
        if not m or m.group(1) == m.group(2):
            continue
        # Layout: facelab_london/morph_<method>/<a>_<b>.jpg (methods: amsl, facemorpher, opencv, stylegan, webmorph)
        method = p.parent.name.removeprefix("morph_")
        morphs.append((p, method, m.group(1), m.group(2)))
    return morphs


THREE_QUARTER = ("neutral_left_3quarter", "neutral_right_3quarter", "smiling_left_3quarter", "smiling_right_3quarter")


def build_pairs(morph_root: Path, frll_root: Path) -> list[dict]:
    views = {v: subject_images(frll_root / v) for v in ("neutral_front", "smiling_front", *THREE_QUARTER)}
    if not views["neutral_front"] or not views["smiling_front"]:
        raise SystemExit(f"FRLL neutral_front/smiling_front not found under {frll_root}; run download_frll.py")
    subjects = sorted(set(views["neutral_front"]) & set(views["smiling_front"]))
    rows = []

    def add(reference: Path, ref_view: str, subject: str, probe_views: tuple[str, ...], **extra) -> None:
        for v in probe_views:
            if subject in views[v]:
                rows.append({"reference": str(reference), "probe": str(views[v][subject]), "reference_view": ref_view,
                             "probe_view": v, **extra})

    for s in subjects:
        bona = {"label": "bona_fide", "morph_method": "", "subject_a": s, "subject_b": ""}
        add(views["neutral_front"][s], "neutral_front", s, ("smiling_front", *THREE_QUARTER), **bona)
        add(views["smiling_front"][s], "smiling_front", s, ("neutral_front", *THREE_QUARTER), **bona)
    morphs = find_morphs(morph_root)
    if not morphs:
        sample = [str(p.relative_to(morph_root)) for p in list(morph_root.rglob("*"))[:10]]
        raise SystemExit(f"no morph files with two subject ids found under {morph_root}; first entries: {sample}")
    for path, method, a, b in morphs:
        if a not in subjects or b not in subjects:
            continue
        for probe_subject in (a, b):
            add(path, "morph", probe_subject, ("smiling_front", *THREE_QUARTER),
                label="morph", morph_method=method, subject_a=a, subject_b=b)
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    args = ap.parse_args(argv)
    if not args.skip_download:
        with urllib.request.urlopen(RECORD, timeout=60) as resp:
            files = {f["key"]: f for f in json.load(resp)["files"]}
        md5 = None
        with urllib.request.urlopen(files["MD5SUM.txt"]["links"]["self"], timeout=60) as resp:
            for line in resp.read().decode().splitlines():
                if "FRLL-Morphs.tar.gz" in line:
                    md5 = line.split()[0]
        f = files["FRLL-Morphs.tar.gz"]
        extract(download(f["links"]["self"], DEST / "archives" / f["key"], expected_size=f["size"], md5=md5), DEST)
    rows = build_pairs(DEST, FRLL)
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS / "m2_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[(r["label"], r["morph_method"])] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"wrote {SPLITS / 'm2_pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
