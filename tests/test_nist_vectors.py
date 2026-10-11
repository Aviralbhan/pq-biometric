"""NIST ACVP known-answer tests for ML-KEM-768 and ML-DSA-65.

Vectors come from usnistgov/ACVP-Server via scripts/download_acvp.py.
Covered: ML-KEM keyGen (seed d||z -> ek), ML-DSA keyGen (seed -> pk), and
ML-DSA sigVer for the pure external interface (with context) and external mu.
Not covered, because `cryptography` has no API for them: ML-KEM encapsulation
with fixed randomness, decapsulation from an expanded dk, deterministic
ML-DSA signing, and HashML-DSA.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric import mldsa, mlkem

VECTORS = Path(__file__).parent / "vectors"


def load_cases(name: str):
    path = VECTORS / name
    if not path.exists():
        pytest.skip(f"{name} missing; run scripts/download_acvp.py", allow_module_level=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(g, t) for g in data["testGroups"] for t in g["tests"]]


MLKEM_KEYGEN = load_cases("mlkem768_keygen.json")
MLDSA_KEYGEN = load_cases("mldsa65_keygen.json")
MLDSA_SIGVER = load_cases("mldsa65_sigver.json")


def _id(case) -> str:
    return f"tc{case[1]['tcId']}"


@pytest.mark.parametrize("case", MLKEM_KEYGEN, ids=_id)
def test_mlkem768_keygen(case):
    _, t = case
    seed = bytes.fromhex(t["d"]) + bytes.fromhex(t["z"])
    priv = mlkem.MLKEM768PrivateKey.from_seed_bytes(seed)
    assert priv.public_key().public_bytes_raw().hex().upper() == t["ek"]
    # The derived key must also decapsulate what its own public key encapsulates.
    ss, ct = mlkem.MLKEM768PublicKey.from_public_bytes(bytes.fromhex(t["ek"])).encapsulate()
    assert priv.decapsulate(ct) == ss


@pytest.mark.parametrize("case", MLDSA_KEYGEN, ids=_id)
def test_mldsa65_keygen(case):
    _, t = case
    priv = mldsa.MLDSA65PrivateKey.from_seed_bytes(bytes.fromhex(t["seed"]))
    assert priv.public_key().public_bytes_raw().hex().upper() == t["pk"]


@pytest.mark.parametrize("case", MLDSA_SIGVER, ids=_id)
def test_mldsa65_sigver(case):
    g, t = case
    pub = mldsa.MLDSA65PublicKey.from_public_bytes(bytes.fromhex(t["pk"]))
    sig = bytes.fromhex(t["signature"])
    try:
        if g.get("externalMu"):
            pub.verify_mu(sig, bytes.fromhex(t["mu"]))
        else:
            pub.verify(sig, bytes.fromhex(t["message"]), context=bytes.fromhex(t.get("context", "")))
        passed = True
    except (InvalidSignature, ValueError):
        passed = False
    assert passed is t["testPassed"], t.get("reason")


def test_vector_counts():
    assert len(MLKEM_KEYGEN) >= 25 and len(MLDSA_KEYGEN) >= 25 and len(MLDSA_SIGVER) >= 30
    assert any(t["testPassed"] for _, t in MLDSA_SIGVER) and not all(t["testPassed"] for _, t in MLDSA_SIGVER)
