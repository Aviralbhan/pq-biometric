"""CelebA-Spoof has no direct download: it is on Google Drive / Baidu behind a usage agreement.

    python scripts/download_celeba_spoof.py            # prints the steps and checks what is in place

1. Read and accept the dataset agreement (non-commercial research only, no
   redistribution): https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-agreement
2. Download the Google Drive folder (link in that README) and unzip so that
   data/celeba_spoof/CelebA_Spoof/Data/{train,test}/ and .../metas/intra_test/ exist.
3. Run python scripts/filter_celeba_spoof.py to build the train/val/test CSVs.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from settings import ROOT  # noqa: E402

ROOT_DIR = ROOT / "data" / "celeba_spoof" / "CelebA_Spoof"
NEEDED = ["Data/train", "Data/test", "metas/intra_test/train_label.json", "metas/intra_test/test_label.json"]


def main() -> int:
    print(__doc__)
    missing = [p for p in NEEDED if not (ROOT_DIR / p).exists()]
    for p in NEEDED:
        print(f"  [{'x' if p not in missing else ' '}] {ROOT_DIR / p}")
    return 1 if missing else 0


if __name__ == "__main__":
    raise SystemExit(main())
