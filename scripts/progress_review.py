"""Build docs/PROGRESS_REVIEW.md. Every number comes from a CSV in eval/results/ (none typed by hand).

    python scripts/progress_review.py

Tables whose CSV does not exist yet are marked "pending". Status, limitations and next
steps are text kept in this script; update them here.
"""

from __future__ import annotations

import csv
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from settings import load_config  # noqa: E402

RESULTS = ROOT / "eval" / "results"
OUT = ROOT / "docs" / "PROGRESS_REVIEW.md"

STATUS = [
    ("S0 Repo, config, CI, Docker files", "done", "Docker image not built yet (no Docker on the dev laptop)"),
    ("S1 Crypto suite (ML-KEM-768, ML-DSA-65; X25519, Ed25519)", "done", "NIST ACVP vectors pass"),
    ("S2 Server and checks 1-6", "done", "every reason code tested in both modes"),
    ("S3 Matcher (ArcFace) on FRLL", "done", "table below"),
    ("S4 Datasets and preprocessing", "partly", "FRLL, FRLL-Morphs, OpenForensics in; CelebA-Spoof not yet"),
    ("S5 Zero-shot M1 / M3", "partly", "M3 done (Val; Test-Dev overnight); M1 waits for CelebA-Spoof"),
    ("S6 Capture client", "partly", "protocol verified live in both modes; acceptance under final thresholds pending"),
    ("S7 M1 / M2 / M3 training", "partly", "M2 done; M3 LayerNorm-tuned; M1 fine-tune waits for CelebA-Spoof"),
    ("S8 M5 recordings", "partly", "10 bona fide sessions; OBS sessions need OBS installed"),
    ("S9 M4 and replay suite", "partly", "code done, live M4 works; 1,000-run suite with real models pending"),
    ("S10 ONNX export", "done", "PyTorch vs ONNX parity table below"),
    ("S11 Integration and calibration", "partly", "real models in the server; thresholds preliminary"),
    ("S12 Evaluation (latency, throughput, attack matrix)", "pending", "crypto micro-benchmark only"),
    ("S13 Report and reproducibility", "pending", ""),
]

LIMITATIONS = [
    "M1 runs the pretrained MiniFASNet with its own argmax rule: no CelebA-Spoof data yet, so no M1 numbers.",
    "M5 runs only its virtual-camera name rule; the classifier needs ~40-60 sessions per class (have 10 bona fide, 0 OBS).",
    "Thresholds are preliminary (validation data only, never webcam captures); final calibration is S11.",
    "FRLL is studio-quality and London-based, CelebA-Spoof celebrity-heavy: performance on Indian faces and webcams is untested.",
    "M2 thresholds come from cross-validation on FRLL-Morphs (no separate morph test set); morph generators outside FRLL-Morphs are untested.",
    "Runs on Windows with an in-process Redis (fakeredis); the Docker/Linux setup and latency measurements are S12 work.",
]

NEXT = [
    ("by 18 Oct", "CelebA-Spoof: M1 zero-shot at three operating points, then fine-tuning and re-export (S5, S7, S10)"),
    ("by 20 Oct", "Replay suite with real models, 1,000 runs per attack per mode (S9)"),
    ("19-25 Oct", "M5 recordings across the team: webcam, OBS video, OBS image (S8)"),
    ("by 1 Nov", "Final calibration on validation data, M5 classifier, Docker build on Linux (S11)"),
    ("2-8 Nov", "Latency per stage, throughput, CPU/memory, attack x mode matrix on an idle Linux machine (S12)"),
    ("9-15 Nov", "Report from the CSVs, fresh-clone rebuild by a teammate, final demo (S13)"),
]


def read(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def pct(x: str | float) -> str:
    return f"{100 * float(x):.2f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def matcher() -> str:
    rows = read("matcher.csv")
    if not rows:
        return "_pending_"
    sel = [r for r in rows if r["view"] == "all"]
    return table(["half", "threshold (FMR 0.1% on val)", "FMR", "FNMR", "genuine / impostor pairs"],
                 [[r["split"], f"{float(r['threshold']):.4f}", pct(r["fmr"]), pct(r["fnmr"]), f"{r['n_genuine']} / {r['n_impostor']}"] for r in sel])


def m2() -> str:
    rows = [r for r in read("m2_morph.csv") or [] if r["rows"] == "pooled_held_out"]
    if not rows:
        return "_pending_"
    methods = sorted(k[6:-1] for k in rows[0] if k.startswith("apcer[") and rows[0][k] != "")
    body = [[pct(r["target_bpcer"]), pct(r["bpcer"])] + [pct(r[f"apcer[{m}]"]) for m in methods] for r in rows]
    counts = ", ".join(f"{m} {rows[0][f'n_attack[{m}]']}" for m in methods)
    no_face = ", ".join(f"{m} {rows[0].get(f'no_face[{m}]', '?')}" for m in methods)
    return (table(["BPCER target", "BPCER achieved"] + [f"APCER {m}" for m in methods], body)
            + f"\n\nPooled held-out over **{rows[0]['n_bona_fide']} bona fide pairs**; morph pairs: {counts}. "
            f"EER {pct(rows[0]['eer'])}, AUC {float(rows[0]['auc']):.4f}. Images with no face found: {no_face}. "
            f"Thresholds per fold from an inner identity-grouped CV on training subjects only.")


def m3() -> str:
    parts = []
    val = [r for r in read("m3_deepfake_zero_shot.csv") or []]
    if val:
        rows = [[r["split"], pct(r["target_bpcer"]), f"{float(r['threshold']):.4f}", pct(r["bpcer"]), pct(r["apcer_max"]),
                 pct(r["eer"]), f"{float(r['auc']):.4f}", f"{r['n_bona_fide']} / {r.get('n_attack[openforensics_fake]', '?')}"] for r in val]
        parts.append("Zero-shot GenD, thresholds from all of Val. The Val rows are that calibration set itself (BPCER matches its "
                     "target by construction); the Test-Dev rows, when present, are the headline independent result:\n\n"
                     + table(["split", "BPCER target", "threshold", "BPCER", "APCER", "EER", "AUC", "real / fake faces"], rows))
    cmp_rows = read("m3_zero_shot_vs_ln_tuned.csv")
    if cmp_rows:
        parts.append("Test-Dev, zero-shot vs LayerNorm-tuned (both with thresholds from the same 20% Val calibration split):\n\n"
                     + table(["model", "thresholds from", "BPCER target", "BPCER", "APCER", "EER", "AUC"],
                             [[r["model"], r["thresholds_from"], pct(r["target_bpcer"]), pct(r["bpcer"]), pct(r["apcer_max"]),
                               pct(r["eer"]), f"{float(r['auc']):.4f}"] for r in cmp_rows if r["thresholds_from"] != "all Val"]))
    else:
        parts.append("Test-Dev zero-shot vs tuned: _pending (scoring overnight)_.")
    tune = read("m3_ln_tuning.csv")
    if tune:
        parts.append("LayerNorm-only tuning on 80% of Val, measured on the 20% calibration split: "
                     + "; ".join(f"epoch {r['epoch']}: AUC {float(r['calib_auc']):.4f}, EER {pct(r['calib_eer'])}" for r in tune) + ".")
    return "\n\n".join(parts) if parts else "_pending_"


def replay() -> str:
    rows = read("replay.csv")
    if not rows:
        return "_pending: the 1,000-run suite with real models is S9 (the earlier stub run was discarded)._"
    return table(["mode", "case", "runs", "expected", "correct code"],
                 [[r["mode"], r["case"], r["runs"], r["expected_code"], pct(r["correct_code_rate"])] for r in rows])


def crypto() -> str:
    sizes, bench = read("crypto_sizes.csv"), read("crypto_bench.csv")
    if not sizes or not bench:
        return "_pending: benchmark runs overnight on the idle machine._"
    s = {r["mode"]: r for r in sizes}
    size_tab = table(["", "X25519 / Ed25519", "ML-KEM-768 / ML-DSA-65"],
                     [["KEM public key (B)", s["CLASSICAL"]["kem_public_key_B"], s["PQ"]["kem_public_key_B"]],
                      ["KEM ciphertext (B)", s["CLASSICAL"]["kem_ciphertext_B"], s["PQ"]["kem_ciphertext_B"]],
                      ["signature (B)", s["CLASSICAL"]["signature_B"], s["PQ"]["signature_B"]],
                      ["extra per request (B)", "-", str(int(s["PQ"]["per_transaction_asymmetric_B"]) - int(s["CLASSICAL"]["per_transaction_asymmetric_B"]))]])
    b = {(r["mode"], r["op"]): r for r in bench}
    ops = ["encapsulate", "decapsulate", "sign_manifest", "verify_manifest", "sign_verdict", "seal_bundle", "unseal_bundle"]
    time_tab = table(["operation (median of %s runs)" % bench[0]["runs"], "classical (us)", "PQ (us)"],
                     [[op, b[("CLASSICAL", op)]["p50_us"], b[("PQ", op)]["p50_us"]] for op in ops])
    return f"{size_tab}\n\n{time_tab}\n\nMachine: {bench[0]['system']}, cryptography {bench[0]['cryptography']}, payload {bench[0]['payload_kb']} KB."


def onnx() -> str:
    rows = read("onnx_export.csv")
    if not rows:
        return "_pending_"
    return table(["model", "parity inputs", "cosine min (PyTorch vs ONNX)", "pass"],
                 [[r["model"], r["inputs"], f"{float(r['cosine_min']):.6f}", r["pass"]] for r in rows])


def main() -> int:
    cfg = load_config()
    t = cfg["thresholds"]
    doc = f"""# Progress review ({date.today().isoformat()})

AI-based biometric evidence authentication for post-quantum digital identity systems.
Numbers below are generated from `eval/results/*.csv` by `scripts/progress_review.py`.
Live demo: `docs/DEMO.md`. Thresholds in use: `{t['version']}` (preliminary, calibrated on validation data).

## Status

{table(["step", "status", "note"], [[a, b, c] for a, b, c in STATUS])}

## Results

**Matcher** (ArcFace buffalo_l on FRLL, 102 subjects, halves by identity):

{matcher()}

**M2 morph detection** (D-MAD SVM on FRLL + FRLL-Morphs, 5-fold identity-grouped CV):

{m2()}

**M3 deepfake detection** (GenD CLIP ViT-L/14, OpenForensics):

{m3()}

**Replay suite** (protocol attacks, both modes):

{replay()}

**Crypto cost** (sizes and median times per operation):

{crypto()}

**ONNX export parity** (the server runs these files):

{onnx()}

## Known limitations

""" + "\n".join(f"- {x}" for x in LIMITATIONS) + """

## Next steps

""" + table(["when", "what"], [[a, b] for a, b in NEXT]) + "\n"
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
