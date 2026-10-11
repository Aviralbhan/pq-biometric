"""Train and evaluate M2 (D-MAD morph detection) on FRLL + FRLL-Morphs pairs.

    python scripts/download_frll.py && python scripts/download_frll_morphs.py
    python scripts/train_m2.py

Embeds every reference and probe with the ArcFace model in config.yaml (aligned
112x112 crops from the same face preprocessing the server uses), trains the RBF SVM
on the train split, and reports val/test results per morph method.

Writes eval/results/scores_m2_morph.csv, eval/results/m2_morph.csv and the model
to models/weights/m2_morph/dmad_svm.joblib.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import date
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.m2_morph.dmad import dmad_feature, save, train_svm  # noqa: E402
from detectors.matcher.arcface import ArcFaceMatcher  # noqa: E402
from eval.harness import RESULTS, write_csv  # noqa: E402
from eval.metrics import detection_summary, threshold_at_bpcer  # noqa: E402
from preprocessing.faces import FacePreprocessor, YuNetDetector  # noqa: E402
from settings import ROOT, load_config, resolve  # noqa: E402


class Embedder:
    def __init__(self, config: dict):
        p = config["preprocessing"]
        for key, path in (("YuNet", p["yunet_model"]), ("ArcFace", config["detectors"]["matcher"]["model"])):
            if not resolve(path).exists():
                raise SystemExit(f"{key} weights missing at {resolve(path)}; run scripts/fetch_models.py")
        self.pre = FacePreprocessor(YuNetDetector(str(resolve(p["yunet_model"])), p["score_threshold"]), p["min_face_px"])
        self.matcher = ArcFaceMatcher(resolve(config["detectors"]["matcher"]["model"]), config["detectors"]["matcher"]["version"])
        self.cache: dict[str, np.ndarray | None] = {}

    def __call__(self, path: str) -> np.ndarray | None:
        if path not in self.cache:
            img = cv2.imread(path)
            crops = self.pre.process(img) if img is not None else None
            self.cache[path] = self.matcher.embed(crops.arcface112) if crops else None
        return self.cache[path]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(ROOT / "data" / "splits" / "m2_pairs.csv"))
    ap.add_argument("--target-bpcer", type=float, default=0.005)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    embed = Embedder(config)
    with open(args.pairs, encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))
    feats, kept, skipped = [], [], 0
    for r in pairs:
        ref, probe = embed(r["reference"]), embed(r["probe"])
        if ref is None or probe is None:
            skipped += 1
            continue
        feats.append(dmad_feature(ref, probe))
        kept.append(r)
    print(f"{len(kept)} pairs embedded, {skipped} skipped (no face >= {config['preprocessing']['min_face_px']} px)")
    x = np.stack(feats)
    y = np.array([int(r["label"] == "morph") for r in kept])
    split = np.array([r["split"] for r in kept])

    model = train_svm(x[split == "train"], y[split == "train"])
    scores = model.predict_proba(x)[:, 1]
    val_bona = scores[(split == "val") & (y == 0)]
    thr = threshold_at_bpcer(val_bona, args.target_bpcer)

    score_rows = [{"split": r["split"], "label": "attack" if r["label"] == "morph" else "bona_fide",
                   "attack_type": r["morph_method"], "reference": r["reference"], "probe": r["probe"], "score": float(s)}
                  for r, s in zip(kept, scores)]
    write_csv(RESULTS / "scores_m2_morph.csv", score_rows)
    results = []
    for part in ("val", "test"):
        rows = [s for s in score_rows if s["split"] == part]
        attacks: dict[str, list[float]] = {}
        for s in rows:
            if s["label"] == "attack":
                attacks.setdefault(s["attack_type"], []).append(s["score"])
        results.append({"split": part, **detection_summary(attacks, [s["score"] for s in rows if s["label"] == "bona_fide"], thr)})
    write_csv(RESULTS / "m2_morph.csv", results)
    path = resolve(config["detectors"]["m2_morph"]["model"])
    save(model, path, f"dmad-svm-frll-{date.today().isoformat()}")
    print(f"model -> {path}")
    for r in results:
        per = ", ".join(f"{k[6:-1]} {v:.3f}" for k, v in r.items() if k.startswith("apcer["))
        print(f"  {r['split']}: BPCER {r['bpcer']:.3f}  EER {r['eer']:.3f}  AUC {r['auc']:.3f}  APCER per method: {per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
