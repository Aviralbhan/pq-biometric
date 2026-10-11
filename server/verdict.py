"""Signed verdict record: the project's novelty.

The server signs one record per consumed challenge, binding the challenge,
transaction, evidence hash, every detector score, the model and threshold
versions, and the decision. The signature is over SHA-384(canonical JSON), with
the server's verdict key for the challenge's mode (ML-DSA-65 in PQ mode).
Only hashes are stored, never face images.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

from crypto import CryptoSuite, Mode, canonical_json, sha384

RECORD_VERSION = "BEA-verdict-v1"


def signer_id(public_key: bytes) -> str:
    return hashlib.sha256(public_key).hexdigest()[:16]


def _clean(score: float | None) -> float | None:
    if score is None or not math.isfinite(score):
        return None
    return round(float(score), 6)


def build_record(
    *,
    challenge_id: str,
    mode: str,
    tx_hash: str,
    sample_hash: str | None,
    scores: dict[str, float | None],
    model_versions: dict[str, str],
    thresholds_version: str,
    decision: str,
    reason_code: str | None,
    timestamp_ms: int,
    signer: str,
) -> dict[str, Any]:
    return {
        "record_version": RECORD_VERSION,
        "challenge_id": challenge_id,
        "mode": mode,
        "tx_hash": tx_hash,
        "sample_hash": sample_hash,
        "scores": {k: _clean(v) for k, v in scores.items()},
        "model_versions": dict(model_versions),
        "thresholds_version": thresholds_version,
        "decision": decision,
        "reason_code": reason_code,
        "timestamp_ms": timestamp_ms,
        "signer": signer,
    }


def sign_record(mode: Mode, private_key: Any, record: dict[str, Any]) -> tuple[bytes, bytes]:
    data = canonical_json(record)
    return data, CryptoSuite(mode).sign(private_key, sha384(data))


def verify_record(mode: Mode, public_key: bytes, record_bytes: bytes, signature: bytes) -> bool:
    return CryptoSuite(mode).verify(public_key, signature, sha384(record_bytes))
