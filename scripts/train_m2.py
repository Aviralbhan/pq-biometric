"""S7: train and evaluate M2 (D-MAD morph detection) on FRLL + FRLL-Morphs with identity-grouped 5-fold CV.

    python scripts/download_frll.py && python scripts/download_frll_morphs.py
    python scripts/train_m2.py [--max-morph-train 6000]

Protocol (CLAUDE.md, M2):
- Every reference and probe is embedded with the ArcFace model in config.yaml, using
  the server's face preprocessing.
- Subjects are split into 5 folds with GroupKFold, so every subject is tested once.
  For the fold that tests subjects T: test = bona fide pairs of T and morphs whose two
  subjects are both in T; train = pairs that involve no subject of T. Morphs mixing a
  test and a training subject are used in neither (no identity leaks across).
- Training morph pairs are subsampled to --max-morph-train per fold (stratified by
  method) to keep the RBF SVM tractable; the SVM uses balanced class weights.
- Out-of-fold scores are pooled; the threshold is set at BPCER 1% on the pooled bona
  fide scores, and APCER per morph method is reported at that threshold. The threshold
  is chosen on the same pooled scores it is reported on (there is no separate
  validation set); the report says so.
- The deployed model is retrained on all subjects.

Writes eval/results/scores_m2_morph.csv (every out-of-fold pair), eval/results/m2_images_without_face.csv,
eval/results/m2_morph.csv
(pooled and per-fold rows, each with its exact bona fide and attack counts) and the
model to models/weights/m2_morph/dmad_svm.joblib.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from datetime import date
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.m2_morph.dmad import dmad_feature, save, svm_score, train_svm  # noqa: E402
from detectors.matcher.arcface import ArcFaceMatcher  # noqa: E402
from eval.harness import RESULTS, rel_path, write_csv  # noqa: E402
from eval.metrics import detection_summary, threshold_at_bpcer  # noqa: E402
from preprocessing.faces import FacePreprocessor, YuNetDetector  # noqa: E402
from settings import ROOT, load_config, resolve  # noqa: E402

TARGET_BPCER = 0.01  # M2 only; see CLAUDE.md


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


def fold_of_subjects(subjects: list[str], n_folds: int, seed: int) -> dict[str, int]:
    gkf = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    out = {}
    for k, (_, test_idx) in enumerate(gkf.split(subjects, groups=subjects)):
        for i in test_idx:
            out[subjects[i]] = k
    return out


def subsample_morphs(idx: np.ndarray, methods: np.ndarray, cap: int, rng: np.random.Generator) -> np.ndarray:
    if len(idx) <= cap:
        return idx
    keep = []
    uniq = sorted(set(methods[idx]))
    for m in uniq:
        sel = idx[methods[idx] == m]
        keep.append(rng.choice(sel, size=min(len(sel), cap // len(uniq)), replace=False))
    return np.concatenate(keep)


def summary_row(name: str, scores: np.ndarray, labels: np.ndarray, methods: np.ndarray, thr: float) -> dict:
    attacks = {m: scores[(labels == 1) & (methods == m)].tolist() for m in sorted(set(methods[labels == 1]))}
    return {"rows": name, "target_bpcer": TARGET_BPCER, **detection_summary(attacks, scores[labels == 0].tolist(), thr),
            **{f"n_attack[{m}]": len(v) for m, v in attacks.items()}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(ROOT / "data" / "splits" / "m2_pairs.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--max-morph-train", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    embed = Embedder(config)
    with open(args.pairs, encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))

    t0 = time.perf_counter()
    feats, kept = [], []
    for r in pairs:
        ref, probe = embed(r["reference"]), embed(r["probe"])
        if ref is not None and probe is not None:
            feats.append(dmad_feature(ref, probe))
            kept.append(r)
    no_face = sorted(p for p, e in embed.cache.items() if e is None)
    by_folder: dict[str, list[int]] = {}
    for path, e in embed.cache.items():
        counts = by_folder.setdefault(Path(path).parent.name, [0, 0])
        counts[0] += 1
        counts[1] += e is None
    write_csv(RESULTS / "m2_images_without_face.csv",
              [{"folder": k, "images": v[0], "without_face": v[1]} for k, v in sorted(by_folder.items())])
    print(f"embedded {len(embed.cache)} images in {time.perf_counter() - t0:.0f} s; {len(no_face)} without a face; "
          f"{len(kept)}/{len(pairs)} pairs kept")
    x = np.stack(feats)
    y = np.array([int(r["label"] == "morph") for r in kept])
    methods = np.array([r["morph_method"] for r in kept])
    subj_a = np.array([r["subject_a"] for r in kept])
    subj_b = np.array([r["subject_b"] or r["subject_a"] for r in kept])

    subjects = sorted(set(subj_a) | set(subj_b))
    fold = fold_of_subjects(subjects, args.folds, args.seed)
    fa, fb = np.array([fold[s] for s in subj_a]), np.array([fold[s] for s in subj_b])
    rng = np.random.default_rng(args.seed)
    oof = np.full(len(kept), np.nan)
    test_fold = np.full(len(kept), -1)
    for k in range(args.folds):
        test = (fa == k) & (fb == k)
        train = (fa != k) & (fb != k)
        tr_idx = np.concatenate([np.where(train & (y == 0))[0], subsample_morphs(np.where(train & (y == 1))[0], methods, args.max_morph_train, rng)])
        t = time.perf_counter()
        model = train_svm(x[tr_idx], y[tr_idx], seed=args.seed)
        oof[test] = svm_score(model, x[test])
        test_fold[test] = k
        print(f"  fold {k}: train {np.sum(y[tr_idx] == 0)} bona fide + {np.sum(y[tr_idx] == 1)} morph, "
              f"test {np.sum(test & (y == 0))} bona fide + {np.sum(test & (y == 1))} morph ({time.perf_counter() - t:.0f} s)")

    tested = test_fold >= 0
    thr = threshold_at_bpcer(oof[tested & (y == 0)], TARGET_BPCER)
    results = [summary_row("pooled_out_of_fold", oof[tested], y[tested], methods[tested], thr)]
    for k in range(args.folds):
        sel = test_fold == k
        results.append(summary_row(f"fold_{k}", oof[sel], y[sel], methods[sel], thr))
    write_csv(RESULTS / "m2_morph.csv", results)
    write_csv(RESULTS / "scores_m2_morph.csv", [
        {"split": "oof", "fold": int(test_fold[i]), "label": "attack" if y[i] else "bona_fide", "attack_type": methods[i],
         "reference_view": kept[i]["reference_view"], "probe_view": kept[i]["probe_view"],
         "reference": rel_path(kept[i]["reference"]), "probe": rel_path(kept[i]["probe"]), "score": float(oof[i])}
        for i in np.where(tested)[0]
    ])

    final_idx = np.concatenate([np.where(y == 0)[0], subsample_morphs(np.where(y == 1)[0], methods, args.max_morph_train, rng)])
    path = resolve(config["detectors"]["m2_morph"]["model"])
    save(train_svm(x[final_idx], y[final_idx], seed=args.seed), path, f"dmad-svm-frll-{date.today().isoformat()}")
    print(f"model -> {path}")
    p = results[0]
    print(f"pooled out-of-fold: threshold {thr:.4f} at BPCER {TARGET_BPCER:.0%} on {int(p['n_bona_fide'])} bona fide pairs; "
          f"BPCER {p['bpcer']:.4f}  EER {p['eer']:.4f}  AUC {p['auc']:.4f}")
    for k, v in p.items():
        if k.startswith("apcer["):
            m = k[6:-1]
            print(f"  APCER[{m}] {v:.4f}  ({p[f'n_attack[{m}]']} morph pairs, {int(p['n_bona_fide'])} bona fide pairs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
