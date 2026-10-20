"""Figures for the course report, drawn from eval/results/*.csv only.

    python docs/report/make_figures.py

Writes docs/report/figures/fig*.png. Figures whose CSV does not exist yet are skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from matplotlib.ticker import FuncFormatter, NullFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "eval" / "results"
OUT = Path(__file__).resolve().parent / "figures"
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "savefig.dpi": 220})

BLUE, GREEN, ORANGE, GREY, RED = "#dbe8f6", "#e2f0d9", "#fde9d9", "#eeeeee", "#c00000"


def rows(name: str) -> list[dict]:
    path = RESULTS / name
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def box(ax, x, y, w, h, lines, color, size=9.5, bold_first=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                fc=color, ec="#333333", lw=0.9))
    if bold_first:
        ax.text(x + w / 2, y + h - 0.02, lines[0], ha="center", va="top", fontsize=size + 0.5, weight="bold")
        ax.text(x + w / 2, y + h - 0.075, "\n".join(lines[1:]), ha="center", va="top", fontsize=size, linespacing=1.35)
    else:
        ax.text(x + w / 2, y + h / 2, "\n".join(lines), ha="center", va="center", fontsize=size, linespacing=1.3)


def arrow(ax, p, q, text="", style="-|>", color="#333333", off=(0, 0.012), size=9):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=11, lw=1.0, color=color))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, ha="center", va="bottom", fontsize=size)


def fig_architecture() -> None:
    fig, ax = plt.subplots(figsize=(11, 6.6))
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(0, 1)
    ax.axis("off")
    top, bot = 0.86, 0.38
    box(ax, 0.00, bot, 0.20, top - bot, [
        "Layer A: capture client", "shows the 2 actions", "4 s webcam video, 15 fps", "frame timestamps and",
        "camera metadata", "", "manifest = {mode,", "challenge_id, nonce,", "tx_hash, sample_hash,", "timestamp}"], BLUE)
    box(ax, 0.27, bot, 0.20, top - bot, [
        "Layer B: PQ channel", "σ = ML-DSA-65.Sign(", "SHA-384(manifest))", "(c, ss) = ML-KEM-768", ".Encaps(pk_server)",
        "K = HKDF-SHA384(ss,", "salt = nonce, info)", "AES-256-GCM(K, video ‖", "meta ‖ manifest ‖ σ)"], GREEN)
    box(ax, 0.54, bot, 0.24, top - bot, [
        "Layer C: verification server", "FastAPI: checks 1–6 in order,", "stop at the first failure", "",
        "then in parallel (ONNX Runtime):", "matcher (ArcFace)", "M1 PAD, M2 morph,", "M3 deepfake, M4 actions,",
        "M5 injection", "", "decision → signed verdict"], ORANGE)
    box(ax, 0.83, bot, 0.17, top - bot, [
        "Layer D: evaluation", "APCER / BPCER", "EER, AUC", "FMR / FNMR", "latency p50 / p95", "CPU, memory",
        "bytes per request", "classical vs PQ", "attack × mode"], GREY)
    box(ax, 0.00, 0.02, 0.20, 0.25, [
        "Enrolment (once)", "device key pair", "(ML-DSA-65 or Ed25519),", "reference photo → ArcFace",
        "template (AES-GCM at rest),", "server KEM key pinned"], GREY)
    box(ax, 0.54, 0.02, 0.11, 0.25, ["Redis", "challenges,", "TTL 60 s,", "atomic", "GETDEL"], GREY)
    box(ax, 0.67, 0.02, 0.11, 0.25, ["SQLite", "signed", "verdicts", "(hashes and", "scores only)"], GREY)
    box(ax, 0.83, 0.02, 0.17, 0.25, ["Auditor (later)", "re-checks a stored", "verdict with the", "server public key"], GREY)
    ax.plot([0.66, 0.66, 0.10], [top, 0.95, 0.95], color="#333333", lw=1.0)
    arrow(ax, (0.10, 0.95), (0.10, top + 0.004))
    ax.text(0.38, 0.958, "(1) challenge: nonce, tx_hash, two actions, mode, server KEM key", ha="center", va="bottom", fontsize=9.5)
    arrow(ax, (0.20, 0.66), (0.27, 0.66), "(2)", size=9)
    arrow(ax, (0.47, 0.66), (0.54, 0.66), "(3)", size=9)
    arrow(ax, (0.54, 0.48), (0.47, 0.48), "(4)", size=9)
    arrow(ax, (0.78, 0.66), (0.83, 0.66), "(5)", size=9)
    arrow(ax, (0.10, 0.27), (0.10, bot))
    arrow(ax, (0.595, bot), (0.595, 0.27), style="<|-|>")
    arrow(ax, (0.725, bot), (0.725, 0.27))
    arrow(ax, (0.78, 0.145), (0.83, 0.145))
    fig.savefig(OUT / "fig1_architecture.png", bbox_inches="tight")
    plt.close(fig)


def fig_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10.5, 3.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    checks = [(["1 Challenge", "valid, unused", "→ GETDEL"], "EXPIRED /\nREPLAY"),
              (["2 Mode", "matches", "challenge"], "MODE_\nMISMATCH"),
              (["3 Decaps,", "HKDF, GCM", "tag valid"], "TAMPER"),
              (["4 Nonce and", "tx_hash", "match"], "TX_\nMISMATCH"),
              (["5 SHA-384", "(video) =", "sample_hash"], "HASH_\nMISMATCH"),
              (["6 Device", "signature", "valid"], "SIG_\nINVALID")]
    w, gap, x0, y, h = 0.095, 0.016, 0.005, 0.55, 0.30
    for i, (c, code) in enumerate(checks):
        x = x0 + i * (w + gap)
        box(ax, x, y, w, h, c, BLUE, size=9, bold_first=False)
        arrow(ax, (x + w / 2, y), (x + w / 2, 0.30), color=RED)
        ax.text(x + w / 2, 0.28, code, ha="center", va="top", fontsize=8.5, color=RED)
        arrow(ax, (x + w, y + h / 2), (x + w + gap, y + h / 2))
    xd = x0 + 6 * (w + gap)
    box(ax, xd, 0.42, 0.14, 0.56, ["in parallel", "matcher", "M1 PAD", "M2 morph", "M3 deepfake", "M4 actions", "M5 injection"],
        ORANGE, size=9)
    arrow(ax, (xd + 0.14, y + h / 2), (xd + 0.16, y + h / 2))
    box(ax, xd + 0.16, 0.42, 0.16, 0.56, ["decision", "any module above", "its threshold →", "reject (first in",
                                           "M5, M3, M1, M2, M4", "order); otherwise", "match ≥ τ → accept"], GREEN, size=8.5)
    fig.savefig(OUT / "fig2_checks_pipeline.png", bbox_inches="tight")
    plt.close(fig)


def fig_crypto() -> None:
    data = rows("crypto_bench.csv")
    if not data:
        return
    ops = ["kem_keygen", "encapsulate", "decapsulate", "sig_keygen", "sign_manifest", "verify_manifest", "sign_verdict"]
    labels = ["KEM\nkeygen", "encaps", "decaps", "sig\nkeygen", "sign\nmanifest", "verify\nmanifest", "sign\nverdict"]
    val = {(r["mode"], r["op"]): float(r["p50_us"]) for r in data}
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xs = range(len(ops))
    ax.bar([x - 0.2 for x in xs], [val[("CLASSICAL", o)] for o in ops], 0.4, label="Classical (X25519 / Ed25519)", color="#7f9fc6")
    ax.bar([x + 0.2 for x in xs], [val[("PQ", o)] for o in ops], 0.4, label="PQ (ML-KEM-768 / ML-DSA-65)", color="#c55a11")
    ax.set_xticks(list(xs), labels)
    ax.set_ylabel("median time per operation (µs)")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.legend(frameon=False, fontsize=9.5)
    ax.grid(axis="y", ls=":", lw=0.6)
    fig.savefig(OUT / "fig3_crypto_ops.png", bbox_inches="tight")
    plt.close(fig)


def fig_m3() -> None:
    data = rows("m3_zero_shot_vs_ln_tuned.csv")
    if not data:
        return
    sel = {(r["model"], r["thresholds_from"], r["target_bpcer"]): float(r["apcer_max"]) * 100 for r in data}
    targets = ["0.005", "0.01", "0.05"]
    zs = [sel[("zero-shot", "Val calib 20%", t)] for t in targets]
    ln = [sel[("LayerNorm-tuned", "Val calib 20%", t)] for t in targets]
    fig, ax = plt.subplots(figsize=(7, 3.3))
    xs = range(3)
    ax.bar([x - 0.2 for x in xs], zs, 0.4, label="zero-shot GenD", color="#7f9fc6")
    ax.bar([x + 0.2 for x in xs], ln, 0.4, label="LayerNorm-tuned GenD", color="#c55a11")
    for x, a, b in zip(xs, zs, ln):
        ax.text(x - 0.2, a + 1, f"{a:.1f}", ha="center", fontsize=9)
        ax.text(x + 0.2, b + 1, f"{b:.1f}", ha="center", fontsize=9)
    ax.set_xticks(list(xs), ["BPCER 0.5%", "BPCER 1%", "BPCER 5%"])
    ax.set_ylabel("APCER on Test-Dev (%)")
    ax.set_ylim(0, 75)
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", lw=0.6)
    fig.savefig(OUT / "fig4_m3_apcer.png", bbox_inches="tight")
    plt.close(fig)


def fig_perf() -> None:
    data = rows("perf_stages.csv")
    if not data:
        return
    want = [("client_seal", "client seal"), ("server_check3_unseal", "unseal\n(check 3)"),
            ("server_check6_signature", "signature\n(check 6)"), ("server_verdict_sign", "verdict\nsignature"),
            ("server_faces", "face\ndetection"), ("server_detectors_parallel", "matcher +\nM1–M5"),
            ("end_to_end", "end to end")]
    p50 = {(r["mode"], r["stage"]): float(r["p50_ms"]) for r in data}
    present = [(k, lab) for k, lab in want if ("PQ", k) in p50 and ("CLASSICAL", k) in p50]
    fig, ax = plt.subplots(figsize=(8, 3.4))
    xs = range(len(present))
    ax.bar([x - 0.2 for x in xs], [p50[("CLASSICAL", k)] for k, _ in present], 0.4, label="Classical", color="#7f9fc6")
    ax.bar([x + 0.2 for x in xs], [p50[("PQ", k)] for k, _ in present], 0.4, label="PQ", color="#c55a11")
    ax.set_xticks(list(xs), [lab for _, lab in present], fontsize=9.5)
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_ylabel("median time (ms)")
    ax.legend(frameon=False)
    ax.grid(axis="y", ls=":", lw=0.6)
    fig.savefig(OUT / "fig5_perf_stages.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (fig_architecture, fig_pipeline, fig_crypto, fig_m3, fig_perf):
        f()
    print("\n".join(sorted(p.name for p in OUT.glob("*.png"))))
