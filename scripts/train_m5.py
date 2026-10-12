"""Train and evaluate M5 (injection) on sessions recorded with scripts/record_m5.py.

    python scripts/train_m5.py [--sessions var/m5_sessions]

Evaluation is leave-one-tool-out: for each virtual-camera tool, train on bona fide +
the other tool(s) and test on the held-out tool, so the score says how well M5
generalises to an injection tool it has not seen. Bona fide sessions are split by
subject (one subject held out per fold when there are several). The device-name rule
is switched off in these folds (the name would give the answer away); the CSV reports
the classifier alone and with the rule, at BPCER 0.5%, 1% and 5% (thresholds from each
fold's training bona fide scores).

Writes eval/results/m5_features.csv, eval/results/scores_m5_injection.csv,
eval/results/m5_injection.csv and the final model (all data) to
models/weights/m5_injection/hgb.joblib.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.m5_injection.classifier import save, train  # noqa: E402
from detectors.m5_injection.features import FEATURE_NAMES, extract, to_vector  # noqa: E402
from eval.harness import RESULTS, write_csv  # noqa: E402
from eval.metrics import operating_point_rows, thresholds_at  # noqa: E402
from preprocessing.video import unpack_jpegs  # noqa: E402
from settings import ROOT, load_config, resolve  # noqa: E402


def load_rows(root: Path, virtual_names: list[str]) -> list[dict]:
    rows = []
    for meta_path in sorted(root.glob("*/*/metadata.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        jpegs = unpack_jpegs((meta_path.parent / "video.beav").read_bytes())
        label = meta.get("label") or meta_path.parent.parent.name
        feats = extract(meta, jpegs, virtual_names)
        rows.append({"session": meta_path.parent.name, "label": label, "subject": meta_path.parent.name.rsplit("_", 1)[0], **feats})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", default=str(ROOT / "var" / "m5_sessions"))
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    rows = load_rows(Path(args.sessions), config["detectors"]["m5_injection"]["virtual_camera_names"])
    tools = sorted({r["label"] for r in rows} - {"bona_fide"})
    if not tools or not any(r["label"] == "bona_fide" for r in rows):
        raise SystemExit("need bona_fide sessions and at least one virtual-camera tool")
    write_csv(RESULTS / "m5_features.csv", rows)

    def xy(subset: list[dict]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([to_vector({**r, "virtual_name": 0.0}) for r in subset])  # rule off: name not a feature
        return x, np.array([int(r["label"] != "bona_fide") for r in subset])

    subjects = sorted({r["subject"] for r in rows if r["label"] == "bona_fide"})
    score_rows, results = [], []
    for held_tool in tools:
        # Bona fide: hold out one subject if there are several, else every 5th session.
        held_subject = subjects[tools.index(held_tool) % len(subjects)] if len(subjects) > 1 else None
        bona = [r for r in rows if r["label"] == "bona_fide"]
        bona_test = [r for i, r in enumerate(bona) if (r["subject"] == held_subject if held_subject else i % 5 == 0)]
        bona_train = [r for r in bona if r not in bona_test]
        train_rows = bona_train + [r for r in rows if r["label"] not in ("bona_fide", held_tool)]
        test_rows = bona_test + [r for r in rows if r["label"] == held_tool]
        if len({r["label"] for r in train_rows}) < 2:
            print(f"  skip {held_tool}: training fold has a single class (record a second tool)")
            continue
        model = train(*xy(train_rows))
        x_train, y_train = xy(train_rows)
        thr = thresholds_at(model.predict_proba(x_train[y_train == 0])[:, 1])  # from this fold's training scores
        x_test, _ = xy(test_rows)
        scores = model.predict_proba(x_test)[:, 1]
        for r, s in zip(test_rows, scores):
            rule = 1.0 if r["virtual_name"] else float(s)
            score_rows.append({"fold": held_tool, "split": "test", "session": r["session"],
                               "label": "bona_fide" if r["label"] == "bona_fide" else "attack",
                               "attack_type": "" if r["label"] == "bona_fide" else r["label"], "score": float(s), "score_with_rule": rule})
        fold = [s for s in score_rows if s["fold"] == held_tool]
        for variant in ("score", "score_with_rule"):
            attacks = {held_tool: [s[variant] for s in fold if s["label"] == "attack"]}
            bona_scores = [s[variant] for s in fold if s["label"] == "bona_fide"]
            results += operating_point_rows(attacks, bona_scores, thr, fold=f"held_out={held_tool}",
                                            variant="classifier" if variant == "score" else "classifier+name_rule")

    write_csv(RESULTS / "scores_m5_injection.csv", score_rows)
    write_csv(RESULTS / "m5_injection.csv", results)
    final = train(*xy(rows))
    path = resolve(config["detectors"]["m5_injection"]["model"])
    save(final, path, f"m5-hgb-{date.today().isoformat()}-n{len(rows)}")
    print(f"model -> {path}")
    for r in results:
        print(f"  {r['fold']:28} {r['variant']:22} BPCER target {r['target_bpcer']:.1%}: APCER {r['apcer_max']:.3f}  BPCER {r['bpcer']:.3f}  EER {r['eer']:.3f}  AUC {r['auc']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
