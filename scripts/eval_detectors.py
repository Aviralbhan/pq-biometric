"""Score dataset images with M1 (PAD) or M3 (deepfake) offline, per attack type.

    python scripts/eval_detectors.py m1_pad --csv data/splits/celeba_spoof_val.csv data/splits/celeba_spoof_test.csv --tag zero_shot
    python scripts/eval_detectors.py m3_deepfake --csv data/splits/m3_val.csv data/splits/m3_test-dev.csv --precropped --tag zero_shot

Input CSVs need columns path, label (bona_fide|attack), attack_type, split. Faces are
found and cropped with the same preprocessing as the server. --precropped (M3) uses
the image itself as the 224x224 GenD crop (download_openforensics.py already cropped it).

Operating points: thresholds for BPCER 0.5%, 1% and 5% are set on the bona fide scores of
--calib-split (default val) and applied to every split; each row also has AUC and EER.
Scores go to eval/results/scores_<module>_<tag>.csv. --resume reuses any score already in a
scores_<module>*.csv for the same model version and image (split labels always come from
the CSVs given now), so only new images are scored. --model evaluates another ONNX file
(e.g. the LayerNorm-tuned GenD) with --version as its name. M3 scores precropped faces in
batches of --batch; --threads sets ONNX Runtime threads for this run.
Writes eval/results/scores_<module>.csv (every image; calibrate.py reads it) and
eval/results/<module>_<tag>.csv (one row per split, APCER per attack type).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from detectors.base import Sample, Stub  # noqa: E402
from detectors.m1_pad import minifasnet  # noqa: E402
from detectors.m3_deepfake import gend  # noqa: E402
from eval.harness import RESULTS, rel_path, write_csv  # noqa: E402
from eval.metrics import OPERATING_POINTS, operating_point_rows, thresholds_at  # noqa: E402
from preprocessing.faces import CenterFaceStub, FaceCrops, FacePreprocessor  # noqa: E402
from settings import load_config  # noqa: E402

BUILDERS = {"m1_pad": minifasnet.build, "m3_deepfake": gend.build}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module", choices=list(BUILDERS))
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--precropped", action="store_true")
    ap.add_argument("--calib-split", default="val", help="split whose bona fide scores set the thresholds")
    ap.add_argument("--limit", type=int, help="max images per CSV")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--model", help="ONNX file to evaluate instead of the one in config.yaml")
    ap.add_argument("--version", help="model version name to record with --model")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    overrides: dict = {"detectors": {"allow_stubs": False}}
    if args.threads:
        overrides["detectors"]["onnx_threads"] = args.threads
        overrides["detectors"][args.module] = {"threads": args.threads}
    if args.model:
        if not args.version:
            ap.error("--model needs --version")
        key = "models" if args.module == "m1_pad" else "model"
        overrides["detectors"].setdefault(args.module, {}).update({key: args.model, "version": args.version})
    config = load_config(args.config, overrides)
    detector = BUILDERS[args.module](config)
    if isinstance(detector, Stub):
        raise SystemExit(f"{args.module} has no weights; nothing to evaluate")
    pre = FacePreprocessor.from_config(config)

    scores_path = RESULTS / f"scores_{args.module}_{args.tag}.csv"
    cached: dict[str, float] = {}
    if args.resume:
        for path in sorted(RESULTS.glob(f"scores_{args.module}*.csv")):
            with open(path, encoding="utf-8") as fh:
                cached.update({r["path"]: float(r["score"]) for r in csv.DictReader(fh) if r.get("model_version") == detector.version})
        print(f"resuming: {len(cached)} cached scores for {detector.version}")
    rows: list[dict] = []
    timings, no_face, n_new = [], 0, 0
    last_checkpoint = [0]  # by count since the last save, not "multiple of N": a partial batch at the end of a CSV shifts the count
    pending: list[tuple[dict, np.ndarray]] = []

    def flush() -> None:
        nonlocal n_new
        if not pending:
            return
        t = time.perf_counter_ns()
        scores = detector.p_fake([crop for _, crop in pending])
        timings.append((time.perf_counter_ns() - t) / 1e6 / len(pending))
        for (it, _), score in zip(pending, scores):
            rows.append(row(it, float(score)))
        n_new += len(pending)
        pending.clear()
        if n_new - last_checkpoint[0] >= args.batch * 50:
            write_csv(scores_path, rows)  # checkpoint: a killed run resumes from here with --resume
            last_checkpoint[0] = n_new
            print(f"  {n_new} scored (checkpointed)", flush=True)

    def row(it: dict, score: float) -> dict:
        return {"split": it.get("split", ""), "label": it["label"], "attack_type": it.get("attack_type", ""),
                "path": rel_path(it["path"]), "score": score, "model_version": detector.version}

    batched = args.precropped and hasattr(detector, "p_fake")
    for csv_path in args.csv:
        with open(csv_path, encoding="utf-8") as fh:
            items = list(csv.DictReader(fh))[: args.limit]
        flush()  # also write a checkpoint at each CSV boundary
        if n_new:
            write_csv(scores_path, rows)
            last_checkpoint[0] = n_new
        for it in items:
            if rel_path(it["path"]) in cached:
                rows.append(row(it, cached[rel_path(it["path"])]))
                continue
            img = cv2.imread(it["path"])
            if img is None:
                continue
            if batched:
                pending.append((it, cv2.resize(img, (224, 224))))
                if len(pending) == args.batch:
                    flush()
                continue
            if args.precropped:
                stub = CenterFaceStub().detect(img)[0]
                crops = [FaceCrops(face=stub, arcface112=np.zeros((112, 112, 3), np.uint8), gend224=cv2.resize(img, (224, 224)), pad80=[])]
            else:
                c = pre.process(img)
                if c is None:
                    no_face += 1
                    continue
                crops = [c]
            t = time.perf_counter_ns()
            score = detector.score(Sample(jpegs=[], metadata={}, actions=[], crops=crops, best=crops[0]))
            timings.append((time.perf_counter_ns() - t) / 1e6)
            rows.append(row(it, score))
            n_new += 1
        flush()
    if not rows:
        raise SystemExit("no images scored")
    print(f"scored {n_new} new images ({no_face} without a usable face), {np.median(timings) if timings else float('nan'):.1f} ms median per image")

    calib = [r["score"] for r in rows if r["split"] == args.calib_split and r["label"] == "bona_fide"]
    if not calib:
        raise SystemExit(f"no bona fide scores in calibration split {args.calib_split!r}")
    thresholds = thresholds_at(calib, OPERATING_POINTS)
    write_csv(scores_path, rows)
    results = []
    for split in sorted({r["split"] for r in rows}):
        part = [r for r in rows if r["split"] == split]
        attacks: dict[str, list[float]] = {}
        for r in part:
            if r["label"] == "attack":
                attacks.setdefault(r["attack_type"] or "attack", []).append(r["score"])
        bona = [r["score"] for r in part if r["label"] == "bona_fide"]
        if attacks and bona:
            results += operating_point_rows(attacks, bona, thresholds, split=split, tag=args.tag, model_version=detector.version,
                                            threshold_from=f"{args.calib_split} bona fide ({len(calib)})",
                                            independent=split != args.calib_split)
    out = write_csv(RESULTS / f"{args.module}_{args.tag}.csv", results)
    print(f"wrote {out}")
    for r in results:
        print(f"  {r['split']:9} BPCER target {r['target_bpcer']:.1%} thr {r['threshold']:.4f}: BPCER {r['bpcer']:.4f}  "
              f"APCER(max) {r['apcer_max']:.4f}  EER {r['eer']:.4f}  AUC {r['auc']:.4f}  ({r['n_bona_fide']} bona fide)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
