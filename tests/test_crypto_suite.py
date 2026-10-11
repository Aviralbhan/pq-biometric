from __future__ import annotations

import os
from dataclasses import replace

import pytest

from crypto import CryptoSuite, Evidence, Manifest, Mode, SealedBundle, TamperError, sha384_hex, tx_hash
from crypto.suite import IV_LEN, NONCE_LEN, SIZES

CHALLENGE_ID = "c-0001"


@pytest.fixture(params=[Mode.CLASSICAL, Mode.PQ], ids=lambda m: m.value)
def suite(request) -> CryptoSuite:
    return CryptoSuite(request.param)


@pytest.fixture
def keys(suite):
    server_kem = suite.kem_keygen()
    device_sig = suite.sig_keygen()
    return server_kem, suite.kem_public_bytes(server_kem), device_sig, suite.sig_public_bytes(device_sig)


def make_evidence(suite: CryptoSuite, device_sig, nonce: bytes) -> Evidence:
    video = os.urandom(4096)
    manifest = Manifest(
        mode=suite.mode.value,
        challenge_id=CHALLENGE_ID,
        nonce=nonce.hex(),
        tx_hash=tx_hash({"amount": 500, "payee": "acct-42", "type": "payment"}),
        sample_hash=sha384_hex(video),
        timestamp=1_760_000_000_000,
    ).to_bytes()
    signature = suite.sign(device_sig, Manifest.from_bytes(manifest).digest())
    return Evidence(video=video, metadata=b'{"device":"test"}', manifest=manifest, signature=signature)


def flip(data: bytes, index: int = 0) -> bytes:
    return data[:index] + bytes([data[index] ^ 0x01]) + data[index + 1 :]


# ---- round trip ------------------------------------------------------------


def test_seal_unseal_round_trip(suite, keys):
    server_kem, server_pub, device_sig, device_pub = keys
    nonce = os.urandom(NONCE_LEN)
    evidence = make_evidence(suite, device_sig, nonce)

    bundle = suite.seal(server_pub, CHALLENGE_ID, nonce, evidence)
    opened = suite.unseal(server_kem, bundle, nonce)

    assert opened == evidence
    manifest = Manifest.from_bytes(opened.manifest)
    assert suite.verify(device_pub, opened.signature, manifest.digest())
    assert manifest.sample_hash == sha384_hex(opened.video)


def test_kem_shared_secret_matches(suite):
    priv = suite.kem_keygen()
    ct, ss = suite.encapsulate(suite.kem_public_bytes(priv))
    assert suite.decapsulate(priv, ct) == ss
    assert len(ss) == 32


def test_private_key_storage_round_trip(suite):
    kem, sig = suite.kem_keygen(), suite.sig_keygen()
    kem2 = suite.load_kem_private(suite.private_bytes(kem))
    sig2 = suite.load_sig_private(suite.private_bytes(sig))
    assert suite.kem_public_bytes(kem2) == suite.kem_public_bytes(kem)
    assert suite.sig_public_bytes(sig2) == suite.sig_public_bytes(sig)


# ---- sizes -----------------------------------------------------------------


@pytest.mark.parametrize(
    "mode, kem_pub, kem_ct, sig_pub, sig",
    [(Mode.CLASSICAL, 32, 32, 32, 64), (Mode.PQ, 1184, 1088, 1952, 3309)],
)
def test_sizes(mode, kem_pub, kem_ct, sig_pub, sig):
    suite = CryptoSuite(mode)
    kem_priv, sig_priv = suite.kem_keygen(), suite.sig_keygen()
    pub = suite.kem_public_bytes(kem_priv)
    ct, _ = suite.encapsulate(pub)
    assert (len(pub), len(ct)) == (kem_pub, kem_ct)
    assert (len(suite.sig_public_bytes(sig_priv)), len(suite.sign(sig_priv, b"x"))) == (sig_pub, sig)
    assert suite.sizes == SIZES[mode]
    assert (suite.sizes.kem_ct, suite.sizes.sig) == (kem_ct, sig)


def test_pq_overhead_bytes():
    """PQ adds ~4.4 KB per transaction over classical (KEM ct + device signature)."""
    pq, cl = SIZES[Mode.PQ], SIZES[Mode.CLASSICAL]
    assert (pq.kem_ct + pq.sig) - (cl.kem_ct + cl.sig) == 4301
    assert pq.kem_ct + pq.sig == 4397


# ---- tamper / wrong key (check 3 -> TAMPER) --------------------------------


@pytest.fixture
def sealed(suite, keys):
    server_kem, server_pub, device_sig, _ = keys
    nonce = os.urandom(NONCE_LEN)
    bundle = suite.seal(server_pub, CHALLENGE_ID, nonce, make_evidence(suite, device_sig, nonce))
    return bundle, nonce


@pytest.mark.parametrize("index", [0, 100, -1])
def test_flipped_payload_byte_rejected(suite, keys, sealed, index):
    bundle, nonce = sealed
    i = index % len(bundle.payload)
    with pytest.raises(TamperError):
        suite.unseal(keys[0], replace(bundle, payload=flip(bundle.payload, i)), nonce)


def test_flipped_kem_ct_rejected(suite, keys, sealed):
    bundle, nonce = sealed
    with pytest.raises(TamperError):
        suite.unseal(keys[0], replace(bundle, kem_ct=flip(bundle.kem_ct, 5)), nonce)


def test_truncated_kem_ct_rejected(suite, keys, sealed):
    bundle, nonce = sealed
    with pytest.raises(TamperError):
        suite.unseal(keys[0], replace(bundle, kem_ct=bundle.kem_ct[:-1]), nonce)


def test_flipped_iv_rejected(suite, keys, sealed):
    bundle, nonce = sealed
    assert len(bundle.iv) == IV_LEN
    with pytest.raises(TamperError):
        suite.unseal(keys[0], replace(bundle, iv=flip(bundle.iv)), nonce)


def test_wrong_server_key_rejected(suite, sealed):
    bundle, nonce = sealed
    with pytest.raises(TamperError):
        suite.unseal(suite.kem_keygen(), bundle, nonce)


def test_swapped_challenge_id_rejected(suite, keys, sealed):
    """challenge_id is in AAD and HKDF info: moving a bundle to another challenge fails."""
    bundle, nonce = sealed
    with pytest.raises(TamperError):
        suite.unseal(keys[0], replace(bundle, challenge_id="c-0002"), nonce)


def test_wrong_nonce_rejected(suite, keys, sealed):
    bundle, _ = sealed
    with pytest.raises(TamperError):
        suite.unseal(keys[0], bundle, os.urandom(NONCE_LEN))


def test_mode_relabel_rejected():
    """A PQ bundle relabelled CLASSICAL cannot be opened by the classical suite."""
    pq, cl = CryptoSuite(Mode.PQ), CryptoSuite(Mode.CLASSICAL)
    kem = pq.kem_keygen()
    nonce = os.urandom(NONCE_LEN)
    bundle = pq.seal(pq.kem_public_bytes(kem), CHALLENGE_ID, nonce, make_evidence(pq, pq.sig_keygen(), nonce))
    with pytest.raises(ValueError):
        cl.unseal(cl.kem_keygen(), bundle, nonce)  # mode mismatch caught before any crypto
    with pytest.raises(TamperError):
        cl.unseal(cl.kem_keygen(), replace(bundle, mode=Mode.CLASSICAL), nonce)


def test_short_nonce_refused(suite, keys):
    with pytest.raises(ValueError):
        suite.seal(keys[1], CHALLENGE_ID, b"short", Evidence(b"", b"", b"", b""))


# ---- signatures (check 6 -> SIG_INVALID) -----------------------------------


def test_changed_manifest_fails_signature(suite, keys, sealed):
    _, _, _, device_pub = keys
    bundle, nonce = sealed
    evidence = suite.unseal(keys[0], bundle, nonce)
    m = Manifest.from_bytes(evidence.manifest)
    for field, value in [("tx_hash", "00" * 48), ("nonce", "11" * 32), ("sample_hash", "22" * 48), ("timestamp", 1)]:
        changed = replace(m, **{field: value})
        assert not suite.verify(device_pub, evidence.signature, changed.digest()), field


def test_wrong_device_key_fails_signature(suite, keys):
    _, _, device_sig, _ = keys
    sig = suite.sign(device_sig, b"digest")
    other_pub = suite.sig_public_bytes(suite.sig_keygen())
    assert not suite.verify(other_pub, sig, b"digest")


def test_flipped_signature_fails(suite, keys):
    _, _, device_sig, device_pub = keys
    sig = suite.sign(device_sig, b"digest")
    assert suite.verify(device_pub, sig, b"digest")
    assert not suite.verify(device_pub, flip(sig, 10), b"digest")


def test_malformed_public_key_fails_closed(suite, keys):
    _, _, device_sig, device_pub = keys
    sig = suite.sign(device_sig, b"digest")
    assert not suite.verify(device_pub[:-1], sig, b"digest")


def test_cross_mode_key_refused():
    pq, cl = CryptoSuite(Mode.PQ), CryptoSuite(Mode.CLASSICAL)
    with pytest.raises(TypeError):
        pq.sign(cl.sig_keygen(), b"x")
    with pytest.raises(TypeError):
        cl.decapsulate(pq.kem_keygen(), b"\x00" * 32)


def test_sealed_bundle_wire_size(suite, sealed):
    bundle, _ = sealed
    assert isinstance(bundle, SealedBundle)
    assert bundle.wire_size() == len(CHALLENGE_ID) + len(suite.mode.value) + suite.sizes.kem_ct + IV_LEN + len(bundle.payload)
