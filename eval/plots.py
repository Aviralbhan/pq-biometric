"""Figures from the CSVs in eval/results/ (nothing is computed that is not in a CSV).

    python -m eval.plots

Writes PNGs to eval/results/figures/ for whichever inputs exist:
  crypto_bench.csv  -> crypto_ops.png       per-operation p50 latency, CLASSICAL vs PQ
  perf_stages.csv   -> stage_latency.png    server stage p50 latency per mode
  perf_summary.csv  -> pqc_share.png        crypto share of end-to-end time, and bytes per request
  attack_matrix.csv -> attack_matrix.png    rejection rate per attack x mode
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from eval.harness import RESULTS  # noqa: E402

MODE_COLORS = {"CLASSICAL": "#4C78A8", "PQ": "#F58518"}


def read(name: str) -> list[dict] | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def grouped_bars(ax, labels: list[str], series: dict[str, list[float]], ylabel: str, log: bool = False) -> None:
    x = np.arange(len(labels))
    width = 0.8 / max(len(series), 1)
    for i, (mode, values) in enumerate(series.items()):
        ax.bar(x + (i - (len(series) - 1) / 2) * width, values, width, label=mode, color=MODE_COLORS.get(mode))
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)


def crypto_ops(out: Path) -> Path | None:
    rows = read("crypto_bench.csv")
    if not rows:
        return None
    ops = list(dict.fromkeys(r["op"] for r in rows))
    by = {(r["mode"], r["op"]): float(r["p50_us"]) for r in rows}
    fig, ax = plt.subplots(figsize=(9, 4))
    grouped_bars(ax, ops, {m: [by[(m, o)] for o in ops] for m in ("CLASSICAL", "PQ")}, "p50 latency (µs, log)", log=True)
    ax.set_title("Crypto operations")
    return save(fig, out / "crypto_ops.png")


def stage_latency(out: Path) -> Path | None:
    rows = read("perf_stages.csv")
    if not rows:
        return None
    stages = [s for s in dict.fromkeys(r["stage"] for r in rows) if s.startswith("server_") and s != "server_total"]
    by = {(r["mode"], r["stage"]): float(r["p50_ms"]) for r in rows}
    modes = [m for m in ("CLASSICAL", "PQ") if any(k[0] == m for k in by)]
    fig, ax = plt.subplots(figsize=(10, 4))
    grouped_bars(ax, [s.removeprefix("server_") for s in stages], {m: [by.get((m, s), 0.0) for s in stages] for m in modes}, "p50 (ms)")
    ax.set_title("Server stage latency")
    return save(fig, out / "stage_latency.png")


def pqc_share(out: Path) -> Path | None:
    rows = read("perf_summary.csv")
    if not rows:
        return None
    modes = [r["mode"] for r in rows]
    fig, (a, b) = plt.subplots(1, 2, figsize=(9, 3.5))
    a.bar(modes, [100 * float(r["crypto_share_p50"]) for r in rows], color=[MODE_COLORS.get(m) for m in modes])
    a.set_ylabel("crypto share of end-to-end (%, p50)")
    b.bar(modes, [float(r["bytes_per_request_mean"]) / 1024 for r in rows], color=[MODE_COLORS.get(m) for m in modes])
    b.set_ylabel("bytes per request (KiB)")
    for ax in (a, b):
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("PQC overhead")
    return save(fig, out / "pqc_share.png")


def attack_matrix(out: Path) -> Path | None:
    rows = read("attack_matrix.csv")
    if not rows:
        return None
    attacks = list(dict.fromkeys(r["attack"] for r in rows))
    modes = [m for m in ("CLASSICAL", "PQ") if any(r["mode"] == m for r in rows)]
    grid = np.full((len(attacks), len(modes)), np.nan)
    for r in rows:
        grid[attacks.index(r["attack"]), modes.index(r["mode"])] = 100 * float(r["rejection_rate"])
    fig, ax = plt.subplots(figsize=(4 + len(modes), 0.45 * len(attacks) + 1.5))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(modes)), modes)
    ax.set_yticks(range(len(attacks)), attacks)
    for i in range(len(attacks)):
        for j in range(len(modes)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, f"{grid[i, j]:.1f}%", ha="center", va="center", color="white" if grid[i, j] > 60 else "black")
    fig.colorbar(im, ax=ax, label="rejected (%)")
    ax.set_title("Rejection rate by attack and mode")
    return save(fig, out / "attack_matrix.png")


def save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> int:
    out = RESULTS / "figures"
    made = [p for p in (crypto_ops(out), stage_latency(out), pqc_share(out), attack_matrix(out)) if p]
    for p in made:
        print(f"wrote {p}")
    if not made:
        print("no result CSVs found in eval/results/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
