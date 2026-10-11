"""Detection metrics (ISO/IEC 30107-3 vocabulary).

Convention: higher score = more likely an attack; a presentation is classified as
an attack when score > threshold.

- APCER: share of attack presentations accepted as bona fide (score <= threshold),
  computed per attack type (PAI species); report the worst type too.
- BPCER: share of bona fide presentations rejected (score > threshold).
- EER: operating point where APCER == BPCER.
- AUC: probability a random attack scores above a random bona fide (ties count 1/2).
For the matcher (higher = same person) use fmr_fnmr / threshold_at_fmr.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import numpy as np


def _arr(x: Iterable[float]) -> np.ndarray:
    a = np.asarray(list(x), dtype=np.float64)
    if a.size == 0:
        raise ValueError("empty score list")
    return a


def apcer(attack_scores: Iterable[float], threshold: float) -> float:
    return float(np.mean(_arr(attack_scores) <= threshold))


def bpcer(bona_fide_scores: Iterable[float], threshold: float) -> float:
    return float(np.mean(_arr(bona_fide_scores) > threshold))


def apcer_per_type(attack_scores_by_type: Mapping[str, Iterable[float]], threshold: float) -> dict[str, float]:
    out = {k: apcer(v, threshold) for k, v in attack_scores_by_type.items()}
    out["max"] = max(out.values())
    return out


def threshold_at_bpcer(bona_fide_scores: Iterable[float], target: float) -> float:
    """Smallest threshold whose BPCER on these scores is <= target."""
    s = np.sort(_arr(bona_fide_scores))
    allowed = int(np.floor(target * s.size))  # bona fide samples allowed above the threshold
    return float(s[s.size - 1 - allowed])


def roc(attack_scores: Iterable[float], bona_fide_scores: Iterable[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, APCER, BPCER) at every distinct score."""
    a, b = _arr(attack_scores), _arr(bona_fide_scores)
    thr = np.unique(np.concatenate([a, b, [-np.inf]]))
    ap = np.array([np.mean(a <= t) for t in thr])
    bp = np.array([np.mean(b > t) for t in thr])
    return thr, ap, bp


def eer(attack_scores: Iterable[float], bona_fide_scores: Iterable[float]) -> tuple[float, float]:
    """(EER, threshold) at the operating point where APCER and BPCER are closest."""
    thr, ap, bp = roc(attack_scores, bona_fide_scores)
    i = int(np.argmin(np.abs(ap - bp)))
    return float((ap[i] + bp[i]) / 2), float(thr[i])


def auc(attack_scores: Iterable[float], bona_fide_scores: Iterable[float]) -> float:
    a, b = _arr(attack_scores), _arr(bona_fide_scores)
    scores = np.concatenate([a, b])
    order = scores.argsort(kind="mergesort")
    ranks = np.empty(scores.size)
    ranks[order] = np.arange(1, scores.size + 1)
    for v in np.unique(scores):  # average ranks of ties
        idx = scores == v
        ranks[idx] = ranks[idx].mean()
    return float((ranks[: a.size].sum() - a.size * (a.size + 1) / 2) / (a.size * b.size))


def fmr_fnmr(genuine_sims: Iterable[float], impostor_sims: Iterable[float], threshold: float) -> tuple[float, float]:
    """Matcher: accept when similarity >= threshold."""
    g, i = _arr(genuine_sims), _arr(impostor_sims)
    return float(np.mean(i >= threshold)), float(np.mean(g < threshold))


def threshold_at_fmr(impostor_sims: Iterable[float], target: float) -> float:
    """Smallest similarity threshold with FMR <= target (accept when sim >= threshold)."""
    s = np.sort(_arr(impostor_sims))
    allowed = int(np.floor(target * s.size))
    return float(np.nextafter(s[s.size - 1 - allowed], np.inf))


def detection_summary(
    attack_scores_by_type: Mapping[str, Iterable[float]], bona_fide_scores: Iterable[float], threshold: float
) -> dict[str, float]:
    """One row of a per-module results CSV."""
    all_attacks = np.concatenate([_arr(v) for v in attack_scores_by_type.values()])
    bona = _arr(bona_fide_scores)
    e, e_thr = eer(all_attacks, bona)
    row = {
        "threshold": threshold,
        "bpcer": bpcer(bona, threshold),
        "apcer_max": apcer_per_type(attack_scores_by_type, threshold)["max"],
        "eer": e,
        "eer_threshold": e_thr,
        "auc": auc(all_attacks, bona),
        "n_bona_fide": float(bona.size),
        "n_attack": float(all_attacks.size),
    }
    for k, v in attack_scores_by_type.items():
        row[f"apcer[{k}]"] = apcer(v, threshold)
    return row
