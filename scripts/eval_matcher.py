"""S3 matcher evaluation on FRLL: enrol neutral_front, probe smiling_front and the 3/4 views.

    python scripts/download_frll.py && python scripts/eval_matcher.py

Genuine pairs: same subject; impostor pairs: every enrolment against every other
subject's probes. Subjects are split in half (val/test) by id: the threshold is set at
FMR 0.1% on val impostors, and FMR/FNMR are reported on both halves.
Writes eval/results/matcher_scores.csv and eval/results/matcher.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import RESULTS, write_csv  # noqa: E402
from eval.metrics import fmr_fnmr, threshold_at_fmr  # noqa: E402
from scripts.download_frll_morphs import subject_images  # noqa: E402
from scripts.train_m2 import Embedder  # noqa: E402
from settings import ROOT, load_config  # noqa: E402

PROBES = ("smiling_front", "neutral_left_3quarter", "neutral_right_3quarter", "smiling_left_3quarter", "smiling_right_3quarter")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frll", default=str(ROOT / "data" / "frll"))
    ap.add_argument("--target-fmr", type=float, default=0.001)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    embed = Embedder(config)
    root = Path(args.frll)

    enrol_all = {s: embed(str(p)) for s, p in subject_images(root / "neutral_front").items()}
    enrol = {s: e for s, e in enrol_all.items() if e is not None}
    probes: list[tuple[str, str, np.ndarray]] = []
    no_face = {"neutral_front": len(enrol_all) - len(enrol)}
    for view in PROBES:
        if (root / view).exists():
            images = subject_images(root / view)
            for s, p in images.items():
                e = embed(str(p))
                if e is not None:
                    probes.append((s, view, e))
            no_face[view] = len(images) - sum(v == view for _, v, _ in probes)
    subjects = sorted(enrol)
    split = {s: ("val" if i % 2 == 0 else "test") for i, s in enumerate(subjects)}

    rows = []
    for s_enrol, e in enrol.items():
        for s_probe, view, p in probes:
            if split.get(s_probe) != split[s_enrol]:
                continue
            rows.append({"split": split[s_enrol], "enrol": s_enrol, "probe": s_probe, "view": view,
                         "kind": "genuine" if s_enrol == s_probe else "impostor", "similarity": float(np.dot(e, p))})
    write_csv(RESULTS / "matcher_scores.csv", rows)
    thr = threshold_at_fmr([r["similarity"] for r in rows if r["split"] == "val" and r["kind"] == "impostor"], args.target_fmr)
    results = []
    for part in ("val", "test"):
        for view in ("all", *PROBES):
            sel = [r for r in rows if r["split"] == part and (view == "all" or r["view"] == view)]
            g = [r["similarity"] for r in sel if r["kind"] == "genuine"]
            i = [r["similarity"] for r in sel if r["kind"] == "impostor"]
            if g and i:
                fmr, fnmr = fmr_fnmr(g, i, thr)
                results.append({"split": part, "view": view, "threshold": thr, "fmr": fmr, "fnmr": fnmr,
                                "n_genuine": len(g), "n_impostor": len(i), "model_version": embed.matcher.version,
                                "images_without_face": sum(no_face.values()) if view == "all" else no_face.get(view, 0)})
    write_csv(RESULTS / "matcher.csv", results)
    print(f"  images without a face >= {config['preprocessing']['min_face_px']} px: {no_face} (excluded; both halves)")
    for r in results:
        if r["view"] == "all":
            print(f"  {r['split']}: threshold {r['threshold']:.4f} (FMR {args.target_fmr:.1%} on val)  FMR {r['fmr']:.4f}  FNMR {r['fnmr']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
