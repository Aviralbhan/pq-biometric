"""S7: train and evaluate M2 (D-MAD morph detection) on FRLL + FRLL-Morphs with identity-grouped 5-fold CV.

    python scripts/download_frll.py && python scripts/download_frll_morphs.py
    python scripts/train_m2.py

Protocol (CLAUDE.md section 4, M2):
- Every reference and probe is embedded with the ArcFace model in config.yaml, using the
  server's face preprocessing (embeddings cached in var/m2_embeddings.npz, keyed by the
  detector and matcher settings).
- Subjects are split into 5 folds with GroupKFold, so every subject is tested once. For the
  fold that tests subjects T: test = bona fide pairs of T and morphs whose two subjects are
  both in T; train = all pairs that involve no subject of T (no subsampling). Morphs mixing a
  test and a training subject are used in neither.
- RBF SVM with balanced class weights. No leakage in the threshold: in each outer fold the
  thresholds for BPCER 0.5%, 1% and 5% come only from that fold's training subjects, via an
  inner 4-fold GroupKFold over them (held-out inner bona fide scores), and are applied to the
  outer held-out pairs; held-out decisions are pooled. EER and AUC are threshold-free.
  (--threshold-source train uses the SVM's scores on its own training pairs instead; those
  sit at the margin, so all three thresholds collapse to one value. Kept only for comparison.)
- The deployed model is retrained on all pairs; its thresholds come from the pooled outer
  held-out bona fide scores (eval/results/m2_deploy_thresholds.csv; calibrate.py uses 1%).

Writes eval/results/m2_morph.csv (pooled and per-fold rows at each operating point, with
exact bona fide and morph pair counts and the no-face count per method),
eval/results/scores_m2_morph.csv, eval/results/m2_images_without_face.csv,
eval/results/m2_deploy_thresholds.csv and models/weights/m2_morph/dmad_svm.joblib.
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
from eval.metrics import OPERATING_POINTS, auc, eer, operating_point_rows, thresholds_at  # noqa: E402
from preprocessing.faces import FacePreprocessor, YuNetDetector  # noqa: E402
from settings import ROOT, load_config, resolve  # noqa: E402

DEPLOY_TARGET = 0.01  # the operating point calibrate.py uses for M2 (CLAUDE.md)
CACHE = ROOT / "var" / "m2_embeddings.npz"


class Embedder:
    def __init__(self, config: dict):
        p = config["preprocessing"]
        for key, path in (("YuNet", p["yunet_model"]), ("ArcFace", config["detectors"]["matcher"]["model"])):
            if not resolve(path).exists():
                raise SystemExit(f"{key} weights missing at {resolve(path)}; run scripts/fetch_models.py")
        self.pre = FacePreprocessor(YuNetDetector(str(resolve(p["yunet_model"])), p["score_threshold"]), p["min_face_px"],
                                   score_margin=p.get("face_score_margin", 0.1))
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


def inner_bona_scores(x: np.ndarray, y: np.ndarray, sa: np.ndarray, sb: np.ndarray, outer_train: np.ndarray,
                      n_inner: int, seed: int) -> np.ndarray:
    """Held-out bona fide scores from an identity-grouped inner CV over the outer fold's training subjects."""
    subjects = sorted(set(sa[outer_train]) | set(sb[outer_train]))
    inner = fold_of_subjects(subjects, n_inner, seed + 1)
    ia = np.array([inner.get(s, -1) for s in sa])
    ib = np.array([inner.get(s, -1) for s in sb])
    scores = []
    for j in range(n_inner):
        test = outer_train & (ia == j) & (ib == j) & (y == 0)
        train = outer_train & (ia != j) & (ib != j)
        model = train_svm(x[train], y[train], seed=seed)
        scores.append(svm_score(model, x[test]))
    return np.concatenate(scores)


def load_cache(key: str) -> dict[str, np.ndarray | None]:
    if not CACHE.exists():
        return {}
    d = np.load(CACHE, allow_pickle=False)
    if str(d["key"]) != key:
        return {}
    return {str(p): (e if ok else None) for p, e, ok in zip(d["paths"], d["emb"], d["ok"])}


def save_cache(key: str, cache: dict[str, np.ndarray | None]) -> None:
    paths = sorted(cache)
    emb = np.stack([cache[p] if cache[p] is not None else np.zeros(512, np.float32) for p in paths])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, key=key, paths=np.array(paths), emb=emb, ok=np.array([cache[p] is not None for p in paths]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", default=str(ROOT / "data" / "splits" / "m2_pairs.csv"))
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--inner-folds", type=int, default=4)
    ap.add_argument("--threshold-source", choices=["inner_cv", "train"], default="inner_cv")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    p = config["preprocessing"]
    key = "|".join(str(v) for v in (p["yunet_model"], p["score_threshold"], p["min_face_px"], p.get("face_score_margin", 0.1),
                                    config["detectors"]["matcher"]["version"]))
    embed = Embedder(config)
    embed.cache.update(load_cache(key))
    with open(args.pairs, encoding="utf-8") as fh:
        pairs = list(csv.DictReader(fh))

    t0 = time.perf_counter()
    feats, kept = [], []
    for r in pairs:
        ref, probe = embed(r["reference"]), embed(r["probe"])
        if ref is not None and probe is not None:
            feats.append(dmad_feature(ref, probe))
            kept.append(r)
    save_cache(key, embed.cache)
    by_folder: dict[str, list[int]] = {}
    for path, e in embed.cache.items():
        counts = by_folder.setdefault(Path(path).parent.name, [0, 0])
        counts[0] += 1
        counts[1] += e is None
    write_csv(RESULTS / "m2_images_without_face.csv",
              [{"folder": k, "images": v[0], "without_face": v[1]} for k, v in sorted(by_folder.items())])
    no_face_by_method = {k.removeprefix("morph_"): v[1] for k, v in by_folder.items() if k.startswith("morph_")}
    print(f"embedded {len(embed.cache)} images ({time.perf_counter() - t0:.0f} s); without a face: "
          + ", ".join(f"{k} {v[1]}/{v[0]}" for k, v in sorted(by_folder.items())) + f"; {len(kept)}/{len(pairs)} pairs kept", flush=True)

    x = np.stack(feats)
    y = np.array([int(r["label"] == "morph") for r in kept])
    methods = np.array([r["morph_method"] for r in kept])
    subj_a = np.array([r["subject_a"] for r in kept])
    subj_b = np.array([r["subject_b"] or r["subject_a"] for r in kept])
    subjects = sorted(set(subj_a) | set(subj_b))
    fold = fold_of_subjects(subjects, args.folds, args.seed)
    fa, fb = np.array([fold[s] for s in subj_a]), np.array([fold[s] for s in subj_b])
    sa, sb = subj_a, subj_b

    source_label = ("SVM scores on the fold's own training pairs" if args.threshold_source == "train"
                    else f"inner {args.inner_folds}-fold CV on the fold's training subjects")
    oof = np.full(len(kept), np.nan)
    test_fold = np.full(len(kept), -1)
    flagged = {t: np.zeros(len(kept), bool) for t in OPERATING_POINTS}
    fold_rows = []
    for k in range(args.folds):
        test = (fa == k) & (fb == k)
        train = (fa != k) & (fb != k)
        t = time.perf_counter()
        model = train_svm(x[train], y[train], seed=args.seed)
        if args.threshold_source == "train":
            thr = thresholds_at(svm_score(model, x[train & (y == 0)]))
        else:
            thr = thresholds_at(inner_bona_scores(x, y, sa, sb, train, args.inner_folds, args.seed))
        oof[test] = svm_score(model, x[test])
        test_fold[test] = k
        for target, value in thr.items():
            flagged[target][test] = oof[test] > value
        attacks = {m: oof[test & (y == 1) & (methods == m)] for m in sorted(set(methods[test & (y == 1)]))}
        fold_rows += operating_point_rows(attacks, oof[test & (y == 0)], thr, rows=f"fold_{k}", threshold_from=source_label)
        print(f"  fold {k}: train {np.sum(train & (y == 0))} bona fide + {np.sum(train & (y == 1))} morph; "
              f"test {np.sum(test & (y == 0))} + {np.sum(test & (y == 1))}; thresholds "
              + ", ".join(f"{a:.1%}:{b:.4f}" for a, b in thr.items()) + f" ({time.perf_counter() - t:.0f} s)", flush=True)

    tested = test_fold >= 0
    bona_t, morph_t = tested & (y == 0), tested & (y == 1)
    e = eer(oof[morph_t], oof[bona_t])[0]
    a = auc(oof[morph_t], oof[bona_t])
    pooled = []
    for target in OPERATING_POINTS:
        row: dict = {"rows": "pooled_held_out", "threshold_from": source_label, "target_bpcer": target,
                     "threshold": "per fold", "bpcer": float(np.mean(flagged[target][bona_t]))}
        per = {m: float(np.mean(~flagged[target][morph_t & (methods == m)])) for m in sorted(set(methods[morph_t]))}
        row["apcer_max"] = max(per.values())
        row.update({f"apcer[{m}]": v for m, v in per.items()})
        row.update({"eer": e, "auc": a, "n_bona_fide": int(bona_t.sum())})
        row.update({f"n_attack[{m}]": int(np.sum(morph_t & (methods == m))) for m in per})
        row.update({f"no_face[{m}]": no_face_by_method.get(m, 0) for m in per})
        pooled.append(row)
    suffix = "" if args.threshold_source == "inner_cv" else "_train_score_thresholds"
    write_csv(RESULTS / f"m2_morph{suffix}.csv", pooled + fold_rows)
    write_csv(RESULTS / "scores_m2_morph.csv", [
        {"split": "oof", "fold": int(test_fold[i]), "label": "attack" if y[i] else "bona_fide", "attack_type": methods[i],
         "reference_view": kept[i]["reference_view"], "probe_view": kept[i]["probe_view"],
         "reference": rel_path(kept[i]["reference"]), "probe": rel_path(kept[i]["probe"]), "score": float(oof[i])}
        for i in np.where(tested)[0]
    ])

    t = time.perf_counter()
    final = train_svm(x, y, seed=args.seed)
    deploy = thresholds_at(oof[bona_t])
    write_csv(RESULTS / f"m2_deploy_thresholds{suffix}.csv",
              [{"target_bpcer": k, "threshold": v, "from": "pooled 5-fold held-out bona fide scores", "n_bona_fide": int(bona_t.sum())}
               for k, v in deploy.items()])
    path = resolve(config["detectors"]["m2_morph"]["model"])
    if args.threshold_source == "inner_cv":
        save(final, path, f"dmad-svm-frll-{date.today().isoformat()}")
    print(f"final model ({time.perf_counter() - t:.0f} s) -> {path}; deploy threshold at BPCER {DEPLOY_TARGET:.0%}: {deploy[DEPLOY_TARGET]:.4f}")
    print(f"pooled held-out ({int(bona_t.sum())} bona fide pairs): EER {e:.4f}  AUC {a:.4f}")
    for row in pooled:
        per = "  ".join(f"{k[6:-1]} {v:.4f} (n={row['n_attack[' + k[6:-1] + ']']}, no face {row['no_face[' + k[6:-1] + ']']})"
                        for k, v in row.items() if k.startswith("apcer["))
        print(f"  BPCER target {row['target_bpcer']:.1%}: achieved {row['bpcer']:.4f}; APCER {per}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
