"""Score dataset images with M1 (PAD) or M3 (deepfake) offline, per attack type.

    python scripts/eval_detectors.py m1_pad --csv data/splits/celeba_spoof_val.csv data/splits/celeba_spoof_test.csv --tag zero_shot
    python scripts/eval_detectors.py m3_deepfake --csv data/splits/m3_val.csv data/splits/m3_test-dev.csv --precropped --tag zero_shot

Input CSVs need columns path, label (bona_fide|attack), attack_type, split. Faces are
found and cropped with the same preprocessing as the server. --precropped (M3) uses
the image itself as the 224x224 GenD crop (download_openforensics.py already cropped it).

Threshold: BPCER 0.5% on the val bona fide scores (or --threshold).
--resume keeps scores already in scores_<module>.csv for the same model version and
only scores new images (e.g. add Test-Dev after Val). M3 scores precropped faces in
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
from eval.metrics import detection_summary, threshold_at_bpcer  # noqa: E402
from preprocessing.faces import CenterFaceStub, FaceCrops, FacePreprocessor  # noqa: E402
from settings import load_config  # noqa: E402

BUILDERS = {"m1_pad": minifasnet.build, "m3_deepfake": gend.build}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module", choices=list(BUILDERS))
    ap.add_argument("--csv", nargs="+", required=True)
    ap.add_argument("--tag", default="eval")
    ap.add_argument("--precropped", action="store_true")
    ap.add_argument("--threshold", type=float)
    ap.add_argument("--target-bpcer", type=float, default=0.005)
    ap.add_argument("--limit", type=int, help="max images per CSV")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    overrides: dict = {"detectors": {"allow_stubs": False}}
    if args.threads:
        overrides["detectors"]["onnx_threads"] = args.threads
    config = load_config(args.config, overrides)
    detector = BUILDERS[args.module](config)
    if isinstance(detector, Stub):
        raise SystemExit(f"{args.module} has no weights; nothing to evaluate")
    pre = FacePreprocessor.from_config(config)

    scores_path = RESULTS / f"scores_{args.module}.csv"
    rows: list[dict] = []
    if args.resume and scores_path.exists():
        with open(scores_path, encoding="utf-8") as fh:
            rows = [{**r, "score": float(r["score"])} for r in csv.DictReader(fh) if r.get("model_version") == detector.version]
        print(f"resuming: {len(rows)} scores already present for {detector.version}")
    done = {r["path"] for r in rows}  # repo-relative
    timings, no_face, n_new = [], 0, 0
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
        if n_new % (args.batch * 50) == 0:
            print(f"  {n_new} scored", flush=True)

    def row(it: dict, score: float) -> dict:
        return {"split": it.get("split", ""), "label": it["label"], "attack_type": it.get("attack_type", ""),
                "path": rel_path(it["path"]), "score": score, "model_version": detector.version}

    batched = args.precropped and hasattr(detector, "p_fake")
    for csv_path in args.csv:
        with open(csv_path, encoding="utf-8") as fh:
            items = [it for it in list(csv.DictReader(fh))[: args.limit] if rel_path(it["path"]) not in done]
        for it in items:
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

    val_bona = [r["score"] for r in rows if r["split"] == "val" and r["label"] == "bona_fide"]
    thr = args.threshold if args.threshold is not None else threshold_at_bpcer(val_bona or [r["score"] for r in rows if r["label"] == "bona_fide"], args.target_bpcer)
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
            results.append({"split": split, "tag": args.tag, "model_version": detector.version,
                            **detection_summary(attacks, bona, thr)})
    out = write_csv(RESULTS / f"{args.module}_{args.tag}.csv", results)
    print(f"wrote {out}")
    for r in results:
        print(f"  {r['split']}: threshold {r['threshold']:.4f}  BPCER {r['bpcer']:.4f}  APCER(max) {r['apcer_max']:.4f}  EER {r['eer']:.4f}  AUC {r['auc']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
