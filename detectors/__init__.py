"""Detection modules. build_all() returns the matcher and M1-M5 for a config."""

from __future__ import annotations

from typing import Any


def build_all(config: dict[str, Any]) -> dict[str, Any]:
    from detectors.m1_pad import minifasnet
    from detectors.m2_morph import dmad
    from detectors.m3_deepfake import gend
    from detectors.m4_replay import actions
    from detectors.m5_injection import classifier
    from detectors.matcher import arcface

    return {
        "matcher": arcface.build(config),
        "m1_pad": minifasnet.build(config),
        "m2_morph": dmad.build(config),
        "m3_deepfake": gend.build(config),
        "m4_replay": actions.build(config),
        "m5_injection": classifier.build(config),
    }
