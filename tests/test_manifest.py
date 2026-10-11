from __future__ import annotations

import json

import pytest

from crypto import Manifest, canonical_json, sha384, tx_hash


def sample_manifest(**overrides) -> Manifest:
    fields = dict(
        mode="PQ",
        challenge_id="c-0001",
        nonce="ab" * 32,
        tx_hash="cd" * 48,
        sample_hash="ef" * 48,
        timestamp=1_760_000_000_000,
    )
    fields.update(overrides)
    return Manifest(**fields)


def test_canonical_json_is_sorted_and_compact():
    assert canonical_json({"b": 1, "a": [1, 2], "c": "é"}) == '{"a":[1,2],"b":1,"c":"é"}'.encode()


def test_canonical_json_ignores_insertion_order():
    assert canonical_json({"x": 1, "y": 2}) == canonical_json({"y": 2, "x": 1})


def test_canonical_json_rejects_nan():
    with pytest.raises(ValueError):
        canonical_json({"x": float("nan")})


def test_tx_hash_is_order_independent_and_sha384():
    a = tx_hash({"amount": 500, "payee": "acct-42"})
    assert a == tx_hash({"payee": "acct-42", "amount": 500})
    assert len(bytes.fromhex(a)) == 48
    assert a != tx_hash({"amount": 501, "payee": "acct-42"})


def test_manifest_round_trip_and_digest():
    m = sample_manifest()
    data = m.to_bytes()
    assert b" " not in data
    assert Manifest.from_bytes(data) == m
    assert m.digest() == sha384(data) and len(m.digest()) == 48


def test_manifest_digest_changes_with_any_field():
    base = sample_manifest().digest()
    for field, value in [
        ("mode", "CLASSICAL"),
        ("challenge_id", "c-0002"),
        ("nonce", "00" * 32),
        ("tx_hash", "00" * 48),
        ("sample_hash", "00" * 48),
        ("timestamp", 1),
    ]:
        assert sample_manifest(**{field: value}).digest() != base, field


def test_manifest_rejects_extra_or_missing_fields():
    obj = json.loads(sample_manifest().to_bytes())
    with pytest.raises(ValueError):
        Manifest.from_bytes(canonical_json({**obj, "extra": 1}))
    del obj["nonce"]
    with pytest.raises(ValueError):
        Manifest.from_bytes(canonical_json(obj))


def test_manifest_rejects_non_canonical_encoding():
    pretty = json.dumps(json.loads(sample_manifest().to_bytes()), indent=2).encode()
    with pytest.raises(ValueError):
        Manifest.from_bytes(pretty)
