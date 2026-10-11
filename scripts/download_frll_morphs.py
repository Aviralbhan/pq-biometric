"""Download FRLL-Morphs (CC BY-SA 4.0, Zenodo 4415159, ~212 MB) into data/frll_morphs/ and build the M2 pair list.

    python scripts/download_frll.py          # first: the bona fide FRLL images
    python scripts/download_frll_morphs.py

Writes data/splits/m2_pairs.csv with columns
    reference, probe, label (morph|bona_fide), morph_method, subject_a, subject_b, split
Pairs:
    morph:     reference = morph of A and B,         probe = smiling_front of A (and of B)
    bona fide: reference = neutral_front of A,       probe = smiling_front of A
Identities are split train/val/test (60/20/20) by subject id; a morph goes to a split
only if both of its subjects are in that split, so no person crosses splits.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
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
    """(path, method, subject_a, subject_b). The method is the folder under the dataset root."""
    morphs = []
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_EXT:
            continue
        m = PAIR_RE.search(p.stem)
        if not m or m.group(1) == m.group(2):
            continue
        rel = p.relative_to(root).parts
        method = next((part for part in rel[:-1] if part.lower() not in {"frll-morphs", "morphs", "morphed"}), rel[0])
        morphs.append((p, method, m.group(1), m.group(2)))
    return morphs


def assign_splits(subjects: list[str], seed: int = 0) -> dict[str, str]:
    s = sorted(subjects)
    random.Random(seed).shuffle(s)
    n = len(s)
    return {sid: ("train" if i < 0.6 * n else "val" if i < 0.8 * n else "test") for i, sid in enumerate(s)}


def build_pairs(morph_root: Path, frll_root: Path, seed: int = 0) -> list[dict]:
    neutral = subject_images(frll_root / "neutral_front")
    smiling = subject_images(frll_root / "smiling_front")
    if not neutral or not smiling:
        raise SystemExit(f"FRLL neutral_front/smiling_front not found under {frll_root}; run download_frll.py")
    split = assign_splits(sorted(set(neutral) & set(smiling)), seed)
    rows = [
        {"reference": str(neutral[s]), "probe": str(smiling[s]), "label": "bona_fide", "morph_method": "",
         "subject_a": s, "subject_b": "", "split": split[s]}
        for s in sorted(split)
    ]
    morphs = find_morphs(morph_root)
    if not morphs:
        sample = [str(p.relative_to(morph_root)) for p in list(morph_root.rglob("*"))[:10]]
        raise SystemExit(f"no morph files with two subject ids found under {morph_root}; first entries: {sample}")
    for path, method, a, b in morphs:
        if a not in split or b not in split or split[a] != split[b]:
            continue  # cross-split morph: dropped so no identity appears in two splits
        for probe_subject in (a, b):
            rows.append({"reference": str(path), "probe": str(smiling[probe_subject]), "label": "morph", "morph_method": method,
                         "subject_a": a, "subject_b": b, "split": split[a]})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
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
    rows = build_pairs(DEST, FRLL, args.seed)
    SPLITS.mkdir(parents=True, exist_ok=True)
    with open(SPLITS / "m2_pairs.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    counts = defaultdict(int)
    for r in rows:
        counts[(r["split"], r["label"], r["morph_method"])] += 1
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    print(f"wrote {SPLITS / 'm2_pairs.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
