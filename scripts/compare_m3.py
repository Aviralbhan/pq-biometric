"""Side-by-side M3 table on OpenForensics Test-Dev: zero-shot vs LayerNorm-tuned GenD.

    python scripts/compare_m3.py

Reads (all written by scripts/eval_detectors.py):
  m3_deepfake_zero_shot.csv           zero-shot, thresholds from all of Val (headline)
  m3_deepfake_zero_shot_calib20.csv   zero-shot, thresholds from the 20% calib split of Val
  m3_deepfake_ln_tuned.csv            tuned on the 80% tune split, thresholds from the same 20% calib split
Writes eval/results/m3_zero_shot_vs_ln_tuned.csv (Test-Dev rows only; nothing is recomputed).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import RESULTS, write_csv  # noqa: E402

SOURCES = [
    ("zero-shot", "all Val", "m3_deepfake_zero_shot.csv"),
    ("zero-shot", "Val calib 20%", "m3_deepfake_zero_shot_calib20.csv"),
    ("LayerNorm-tuned", "Val calib 20%", "m3_deepfake_ln_tuned.csv"),
]
COLUMNS = ("target_bpcer", "threshold", "bpcer", "apcer_max", "eer", "auc", "n_bona_fide", "n_attack[openforensics_fake]", "model_version")


def main() -> int:
    rows = []
    for model, thresholds_from, name in SOURCES:
        path = RESULTS / name
        if not path.exists():
            print(f"missing {name}; skipped")
            continue
        with open(path, encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["split"] == "test-dev":
                    rows.append({"model": model, "thresholds_from": thresholds_from, **{c: r.get(c, "") for c in COLUMNS}})
    out = write_csv(RESULTS / "m3_zero_shot_vs_ln_tuned.csv", rows)
    print(f"wrote {out}")
    for r in rows:
        print(f"  {r['model']:16} thr from {r['thresholds_from']:14} BPCER target {float(r['target_bpcer']):.1%}: "
              f"BPCER {float(r['bpcer']):.4f}  APCER {float(r['apcer_max']):.4f}  EER {float(r['eer']):.4f}  AUC {float(r['auc']):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
