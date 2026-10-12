"""Set decision thresholds from validation scores and write them into config.yaml.

    python scripts/calibrate.py [--target-bpcer 0.005] [--target-fmr 0.001]
    python scripts/calibrate.py --preliminary --m1-argmax [--m3-tag ln_tuned --m3-split calib]   # before S11 data exists

- M1, M3: threshold at BPCER 0.5% on the val bona fide scores in
  eval/results/scores_<module>_<tag>.csv (--m1-tag / --m3-tag pick the model variant).
- M5: BPCER 0.5% on eval/results/scores_m5_injection.csv (its test folds; no separate val split).
- M2: the BPCER 0.5% threshold of the deployed model (eval/results/m2_deploy_thresholds.csv,
  from its training bona fide scores; scripts/train_m2.py, CLAUDE.md).
- M4: pass/fail, fixed at 0.5.
- Matcher: threshold at FMR 0.1% on val impostors in eval/results/matcher_scores.csv.
A module without a scores file keeps its current threshold and is listed as such.
--m1-argmax: no M1 data yet; M1 uses the pretrained Silent-Face argmax rule (threshold 0.5,
  requires detectors.m1_pad.decision_rule: argmax). M5 without a classifier uses its virtual-camera
  name rule (scores 0 or 1, threshold 0.5).
--preliminary: labels the block "preliminary, calibrated on validation data" (version preliminary-...).
The scores used for M3 must come from the model version configured in detectors.m3_deepfake.

Replaces the thresholds block at the end of config.yaml and writes eval/results/thresholds.csv.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.harness import RESULTS, write_csv  # noqa: E402
from eval.metrics import threshold_at_bpcer, threshold_at_fmr  # noqa: E402
from settings import DEFAULT_CONFIG, load_config  # noqa: E402

MARKER = "# --- thresholds"
SCORED = {"m1_pad": ("val", "scores_m1_pad_{m1}.csv"), "m3_deepfake": ("val", "scores_m3_deepfake_{m3}.csv"),
          "m5_injection": ("test", "scores_m5_injection.csv")}
M2_TARGET = 0.005  # same operating point as every module (CLAUDE.md)


def read(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def render(thresholds: dict, preliminary: bool = False) -> str:
    lines = [f"{MARKER} (written by scripts/calibrate.py; edit by hand only before calibration) ---"]
    if preliminary:
        lines.append("# PRELIMINARY: calibrated on validation data only (never on webcam captures); final calibration is S11.")
    lines.append("thresholds:")
    lines.append(f"  version: {thresholds['version']}")
    for key in ("match", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection"):
        lines.append(f"  {key}: {thresholds[key]!r}")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-bpcer", type=float, default=0.005)
    ap.add_argument("--target-fmr", type=float, default=0.001)
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--m1-tag", default="zero_shot", help="which scores_m1_pad_<tag>.csv to calibrate on")
    ap.add_argument("--m3-tag", default="zero_shot", help="which scores_m3_deepfake_<tag>.csv to calibrate on")
    ap.add_argument("--m3-split", default="val", help="split of that file whose bona fide scores set the M3 threshold")
    ap.add_argument("--m1-argmax", action="store_true", help="M1 on the pretrained argmax rule (threshold 0.5)")
    ap.add_argument("--preliminary", action="store_true")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    new = dict(config["thresholds"])
    report = []
    scored = dict(SCORED)
    scored["m3_deepfake"] = (args.m3_split, scored["m3_deepfake"][1])
    if args.m1_argmax:
        if config["detectors"]["m1_pad"].get("decision_rule") != "argmax":
            raise SystemExit("--m1-argmax needs detectors.m1_pad.decision_rule: argmax in the config")
        scored.pop("m1_pad")
        new["m1_pad"] = 0.5
        report.append({"module": "m1_pad", "threshold": 0.5, "source": "pretrained Silent-Face argmax rule (no M1 data yet)", "n": 0, "target": ""})
    for module, (split, pattern) in scored.items():
        name = pattern.format(m1=args.m1_tag, m3=args.m3_tag)
        rows = read(name)
        if module == "m3_deepfake" and rows:
            versions = {r.get("model_version") for r in rows}
            if versions != {config["detectors"]["m3_deepfake"]["version"]}:
                raise SystemExit(f"{name} is from {versions}, but config runs {config['detectors']['m3_deepfake']['version']}")
        bona = [float(r["score"]) for r in rows or [] if r["split"] == split and r["label"] == "bona_fide"]
        target = args.target_bpcer
        if bona:
            new[module] = round(threshold_at_bpcer(bona, target), 6)
            report.append({"module": module, "threshold": new[module], "source": f"{name}[{split}]", "n": len(bona), "target": f"BPCER {target}"})
        elif module == "m5_injection":
            new[module] = 0.5
            report.append({"module": module, "threshold": 0.5, "source": "virtual-camera name rule (no classifier yet)", "n": 0, "target": ""})
        else:
            report.append({"module": module, "threshold": new[module], "source": "unchanged (no scores)", "n": 0, "target": ""})
    m2 = [r for r in read("m2_deploy_thresholds.csv") or [] if float(r["target_bpcer"]) == M2_TARGET]
    if m2:
        new["m2_morph"] = round(float(m2[0]["threshold"]), 6)
        report.append({"module": "m2_morph", "threshold": new["m2_morph"], "source": "m2_deploy_thresholds.csv", "n": int(m2[0]["n_bona_fide"]), "target": f"BPCER {M2_TARGET}"})
    else:
        report.append({"module": "m2_morph", "threshold": new["m2_morph"], "source": "unchanged (no scores)", "n": 0, "target": ""})
    new["m4_replay"] = 0.5
    report.append({"module": "m4_replay", "threshold": 0.5, "source": "fixed (pass/fail)", "n": 0, "target": ""})
    rows = read("matcher_scores.csv")
    impostors = [float(r["similarity"]) for r in rows or [] if r["split"] == "val" and r["kind"] == "impostor"]
    if impostors:
        new["match"] = round(threshold_at_fmr(impostors, args.target_fmr), 6)
        report.append({"module": "match", "threshold": new["match"], "source": "matcher_scores.csv[val]", "n": len(impostors), "target": f"FMR {args.target_fmr}"})
    else:
        report.append({"module": "match", "threshold": new["match"], "source": "unchanged (no scores)", "n": 0, "target": ""})

    digest = hashlib.sha256(json.dumps({k: new[k] for k in sorted(new) if k != "version"}).encode()).hexdigest()[:8]
    new["version"] = f"{'preliminary-val' if args.preliminary else 'cal'}-{date.today().isoformat()}-{digest}"
    for r in report:
        print(f"  {r['module']:13} {r['threshold']:<10} {r['source']} {r['target']}")
    if args.dry_run:
        return 0
    path = Path(args.config)
    text = path.read_text(encoding="utf-8")
    if MARKER not in text:
        raise SystemExit(f"{path} has no '{MARKER}' block to replace")
    path.write_text(text[: text.index(MARKER)] + render(new, args.preliminary), encoding="utf-8")
    write_csv(RESULTS / "thresholds.csv", [{**r, "version": new["version"]} for r in report])
    print(f"thresholds {new['version']} -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
