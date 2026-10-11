"""Decision rule: any attack module above its threshold rejects; otherwise the matcher decides.

When several modules fire, the reported reason is the first in PRIORITY (capture
provenance first, then content). Every score is still recorded in the verdict.
"""

from __future__ import annotations

from typing import Any

from server.checks import Reason

MODULE_REASONS = {
    "m5_injection": Reason.INJECTION_FAIL,
    "m3_deepfake": Reason.DEEPFAKE_FAIL,
    "m1_pad": Reason.PAD_FAIL,
    "m2_morph": Reason.MORPH_FLAG,
    "m4_replay": Reason.SCRIPT_FAIL,
}
PRIORITY = tuple(MODULE_REASONS)
SCORE_KEYS = ("match", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection")


def decide(scores: dict[str, float], thresholds: dict[str, Any]) -> tuple[bool, Reason | None]:
    for module in PRIORITY:
        if scores[module] > float(thresholds[module]):
            return False, MODULE_REASONS[module]
    if scores["match"] >= float(thresholds["match"]):
        return True, None
    return False, Reason.NO_MATCH
