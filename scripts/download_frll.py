"""Download FRLL (Face Research Lab London Set, CC BY 4.0, figshare 5047666) into data/frll/.

    python scripts/download_frll.py             # neutral/smiling front + 3/4 views (~180 MB)
    python scripts/download_frll.py --all       # also the profile views (~100 MB more)

Used by the matcher (enrol neutral_front, probe smiling_front and 3/4 views) and M2
(bona fide images for the FRLL-Morphs pairs). Do not re-upload the images anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._download import download, extract  # noqa: E402
from settings import ROOT  # noqa: E402

ARTICLE = "https://api.figshare.com/v2/articles/5047666"
DEST = ROOT / "data" / "frll"
CORE = {"neutral_front.zip", "smiling_front.zip", "neutral_left_3quarter.zip", "neutral_right_3quarter.zip",
        "smiling_left_3quarter.zip", "smiling_right_3quarter.zip", "london_faces_info.csv"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args(argv)
    with urllib.request.urlopen(ARTICLE, timeout=60) as resp:
        files = json.load(resp)["files"]
    for f in files:
        if not (args.all or f["name"] in CORE) or f["name"].endswith(".pdf") or f["name"] == "london_faces_ratings.csv":
            continue
        path = download(f["download_url"], DEST / "archives" / f["name"], expected_size=f["size"], md5=f.get("supplied_md5") or None)
        if path.suffix == ".zip":
            extract(path, DEST / path.stem)
        else:
            (DEST / path.name).write_bytes(path.read_bytes())
    print(f"FRLL in {DEST}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
