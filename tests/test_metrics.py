from __future__ import annotations

import numpy as np
import pytest

from eval.metrics import (
    apcer,
    apcer_per_type,
    auc,
    bpcer,
    detection_summary,
    eer,
    fmr_fnmr,
    threshold_at_bpcer,
    threshold_at_fmr,
)


def test_apcer_bpcer():
    assert apcer([0.1, 0.6, 0.9, 0.4], 0.5) == 0.5
    assert bpcer([0.1, 0.6, 0.2, 0.3], 0.5) == 0.25
    assert apcer_per_type({"print": [0.9, 0.8], "screen": [0.1, 0.9]}, 0.5) == {"print": 0.0, "screen": 0.5, "max": 0.5}


def test_threshold_at_bpcer_hits_target():
    bona = np.linspace(0, 1, 1000)
    t = threshold_at_bpcer(bona, 0.005)
    assert bpcer(bona, t) <= 0.005
    assert bpcer(bona, np.nextafter(t, -np.inf)) > 0.005


def test_perfect_and_random_separation():
    assert auc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auc([0.1, 0.2], [0.9, 0.8]) == 0.0
    assert auc([0.5, 0.5], [0.5, 0.5]) == 0.5
    e, _ = eer([0.9, 0.8, 0.7], [0.1, 0.2, 0.3])
    assert e == 0.0


def test_eer_overlapping():
    rng = np.random.default_rng(0)
    attack, bona = rng.normal(1, 1, 5000), rng.normal(-1, 1, 5000)
    e, t = eer(attack, bona)
    assert e == pytest.approx(0.1587, abs=0.01)  # Phi(-1) for unit-variance Gaussians 2 sigma apart
    assert abs(t) < 0.1
    assert auc(attack, bona) == pytest.approx(0.9214, abs=0.01)  # Phi(2 / sqrt(2))


def test_matcher_metrics():
    genuine, impostor = [0.8, 0.7, 0.3], [0.1, 0.2, 0.5]
    assert fmr_fnmr(genuine, impostor, 0.4) == (pytest.approx(1 / 3), pytest.approx(1 / 3))
    t = threshold_at_fmr(np.linspace(-1, 1, 10000), 0.001)
    assert fmr_fnmr([1.0], np.linspace(-1, 1, 10000), t)[0] <= 0.001


def test_summary_row():
    row = detection_summary({"print": [0.9, 0.8], "replay": [0.7, 0.2]}, [0.1, 0.3, 0.4], 0.5)
    assert row["apcer[replay]"] == 0.5 and row["apcer_max"] == 0.5 and row["bpcer"] == 0.0
    assert 0 <= row["eer"] <= 1 and row["n_attack"] == 4


def test_empty_scores_rejected():
    with pytest.raises(ValueError):
        apcer([], 0.5)
