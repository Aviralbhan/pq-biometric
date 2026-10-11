"""S2: every reason code plus a valid request, in both modes, against the real app with stub detectors."""

from __future__ import annotations

import base64
import json
import os
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from client.protocol import ApiClient, DeviceIdentity, ProtocolError, check_challenge, make_manifest, seal_evidence
from crypto import CryptoSuite, Mode, SealedBundle, canonical_json
from detectors.base import Stub
from server import challenge as challenge_mod
from preprocessing.video import encode_jpeg
from server.app import create_app
from server.verdict import verify_record
from tests.conftest import TX, stub_config, synthetic_frames, synthetic_metadata, synthetic_video

MODES = [Mode.PQ, Mode.CLASSICAL]


@pytest.fixture(params=MODES, ids=lambda m: m.value)
def mode(request) -> Mode:
    return request.param


@pytest.fixture
def env(tmp_path):
    app = create_app(stub_config(tmp_path))
    with TestClient(app) as http:
        yield app.state.server, ApiClient(http)


@pytest.fixture
def enrolled(env, mode):
    server, api = env
    ref = encode_jpeg(synthetic_frames(1, seed=99)[0])
    identity = api.enroll("alice", mode, ref)
    return server, api, identity


def reason_of(server, request_id: int) -> str | None:
    return server.db.get_request(request_id)["reason_code"]


def fresh(api: ApiClient, identity: DeviceIdentity, tx=TX):
    ch = api.challenge(identity.user_id, tx, identity.mode)
    check_challenge(identity, ch, tx)
    return ch


def test_valid_request_accepted_and_verdict_signed(enrolled):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    resp, rid = api.verify(seal_evidence(identity, ch, synthetic_video(), synthetic_metadata()))
    assert resp == {"decision": "accepted", "challenge_id": ch["challenge_id"]}
    assert reason_of(server, rid) is None

    mode, record, signature = server.db.get_verdict(ch["challenge_id"])
    assert mode == identity.mode.value
    assert verify_record(identity.mode, identity.verdict_pub, record, signature)
    timing = server.db.get_request(rid)["timing"]
    for stage in ("check1_challenge", "check3_unseal", "check6_signature", "faces", "detectors_parallel", "verdict_sign", "total"):
        assert stage in timing


def test_response_hides_reason(enrolled):
    _, api, identity = enrolled
    ch = fresh(api, identity)
    bundle = seal_evidence(identity, ch, synthetic_video(), synthetic_metadata())
    api.verify(bundle)
    resp, _ = api.verify(bundle)
    assert set(resp) == {"decision", "challenge_id"} and resp["decision"] == "rejected"


def test_replay_exact_resubmit(enrolled):
    server, api, identity = enrolled
    bundle = seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata())
    assert api.verify(bundle)[0]["decision"] == "accepted"
    resp, rid = api.verify(bundle)
    assert resp["decision"] == "rejected" and reason_of(server, rid) == "REPLAY"


def test_expired_challenge(enrolled, monkeypatch):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    bundle = seal_evidence(identity, ch, synthetic_video(), synthetic_metadata())
    server.store.r.delete(f"chal:{ch['challenge_id']}")  # what Redis does when the 60 s TTL runs out
    resp, rid = api.verify(bundle)
    assert resp["decision"] == "rejected" and reason_of(server, rid) == "EXPIRED"


def test_expired_by_clock_even_if_key_survives(enrolled, monkeypatch):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    bundle = seal_evidence(identity, ch, synthetic_video(), synthetic_metadata())
    monkeypatch.setattr(challenge_mod, "now_ms", lambda: ch["expires_ms"] + 1)
    _, rid = api.verify(bundle)
    assert reason_of(server, rid) == "EXPIRED"


def test_unknown_challenge_id(enrolled):
    server, api, identity = enrolled
    bundle = seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata())
    _, rid = api.verify(replace(bundle, challenge_id="0" * 32))
    assert reason_of(server, rid) == "EXPIRED"


def test_mode_mismatch_downgrade(env):
    server, api = env
    ref = encode_jpeg(synthetic_frames(1)[0])
    pq = api.enroll("bob", Mode.PQ, ref)
    classical = api.enroll("bob", Mode.CLASSICAL, ref)
    ch = api.challenge("bob", TX, Mode.PQ)
    # The attacker relabels a PQ challenge and seals classically (key from a classical enrolment).
    downgraded = seal_evidence(classical, {**ch, "mode": "CLASSICAL"}, synthetic_video(), synthetic_metadata())
    _, rid = api.verify(downgraded)
    assert reason_of(server, rid) == "MODE_MISMATCH"
    # Check 1 ran first, so the challenge is now consumed and the genuine PQ bundle is refused too.
    _, rid = api.verify(seal_evidence(pq, ch, synthetic_video(), synthetic_metadata()))
    assert reason_of(server, rid) == "REPLAY"


@pytest.mark.parametrize("where", ["payload", "kem_ct", "iv"])
def test_tamper_flipped_byte(enrolled, where):
    server, api, identity = enrolled
    bundle = seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata())
    data = bytearray(getattr(bundle, where))
    data[len(data) // 2] ^= 0x01
    _, rid = api.verify(replace(bundle, **{where: bytes(data)}))
    assert reason_of(server, rid) == "TAMPER"


def test_garbage_body(env):
    server, api = env
    _, rid = api.verify_bytes(os.urandom(200))
    assert reason_of(server, rid) == "TAMPER"


def test_tx_mismatch_old_nonce(enrolled):
    """Evidence bound to an old challenge, re-sealed under a fresh one by a client holding the device key."""
    server, api, identity = enrolled
    video = synthetic_video()
    old = fresh(api, identity)
    old_manifest = make_manifest(identity, old, video)
    new = fresh(api, identity)
    _, rid = api.verify(seal_evidence(identity, new, video, synthetic_metadata(), manifest=old_manifest))
    assert reason_of(server, rid) == "TX_MISMATCH"


def test_tx_mismatch_swapped_transaction(enrolled):
    server, api, identity = enrolled
    video = synthetic_video()
    ch_a = fresh(api, identity, TX)
    ch_b = fresh(api, identity, {**TX, "payee": "attacker", "amount": 50000})
    manifest_for_a = replace(make_manifest(identity, ch_b, video), tx_hash=ch_a["tx_hash"])
    _, rid = api.verify(seal_evidence(identity, ch_b, video, synthetic_metadata(), manifest=manifest_for_a))
    assert reason_of(server, rid) == "TX_MISMATCH"


def test_hash_mismatch(enrolled):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    manifest = make_manifest(identity, ch, synthetic_video(seed=1))
    _, rid = api.verify(seal_evidence(identity, ch, synthetic_video(seed=2), synthetic_metadata(), manifest=manifest))
    assert reason_of(server, rid) == "HASH_MISMATCH"


def test_sig_invalid_wrong_device_key(enrolled):
    server, api, identity = enrolled
    impostor = replace(identity, sig_private=identity.suite.sig_keygen())
    _, rid = api.verify(seal_evidence(impostor, fresh(api, identity), synthetic_video(), synthetic_metadata()))
    assert reason_of(server, rid) == "SIG_INVALID"


@pytest.mark.parametrize(
    "module, value, reason",
    [
        ("m1_pad", 0.9, "PAD_FAIL"),
        ("m2_morph", 0.9, "MORPH_FLAG"),
        ("m3_deepfake", 0.9, "DEEPFAKE_FAIL"),
        ("m4_replay", 1.0, "SCRIPT_FAIL"),
    ],
)
def test_detector_rejections(enrolled, module, value, reason):
    server, api, identity = enrolled
    server.pipeline.modules[module] = Stub(module, value)
    _, rid = api.verify(seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata()))
    assert reason_of(server, rid) == reason


def test_injection_rule_virtual_camera(enrolled):
    server, api, identity = enrolled
    meta = synthetic_metadata(device="OBS Virtual Camera")
    _, rid = api.verify(seal_evidence(identity, fresh(api, identity), synthetic_video(), meta))
    assert reason_of(server, rid) == "INJECTION_FAIL"


def test_no_match(enrolled):
    server, api, identity = enrolled
    server.thresholds = {**server.thresholds, "match": 1.01}
    _, rid = api.verify(seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata()))
    assert reason_of(server, rid) == "NO_MATCH"


def test_edited_verdict_fails_verification(enrolled):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    api.verify(seal_evidence(identity, ch, synthetic_video(), synthetic_metadata()))
    mode, record, signature = server.db.get_verdict(ch["challenge_id"])
    edited = json.loads(record)
    edited["scores"]["m1_pad"] = 0.0 if edited["scores"]["m1_pad"] else 0.5
    assert not verify_record(identity.mode, identity.verdict_pub, canonical_json(edited), signature)


def test_protocol_rejection_also_gets_a_signed_verdict(enrolled):
    server, api, identity = enrolled
    ch = fresh(api, identity)
    bundle = seal_evidence(identity, ch, synthetic_video(), synthetic_metadata())
    data = bytearray(bundle.payload)
    data[0] ^= 1
    api.verify(replace(bundle, payload=bytes(data)))
    _, record, signature = server.db.get_verdict(ch["challenge_id"])
    rec = json.loads(record)
    assert rec["decision"] == "rejected" and rec["reason_code"] == "TAMPER" and rec["sample_hash"] is None
    assert verify_record(identity.mode, identity.verdict_pub, record, signature)


def test_client_refuses_substituted_server_key(enrolled):
    _, api, identity = enrolled
    ch = api.challenge(identity.user_id, TX, identity.mode)
    other = CryptoSuite(identity.mode)
    ch["server_kem_pub"] = base64.b64encode(other.kem_public_bytes(other.kem_keygen())).decode()
    with pytest.raises(ProtocolError):
        check_challenge(identity, ch, TX)


def test_client_refuses_other_transaction(enrolled):
    _, api, identity = enrolled
    ch = api.challenge(identity.user_id, TX, identity.mode)
    with pytest.raises(ProtocolError):
        check_challenge(identity, ch, {**TX, "amount": 1})


def test_challenge_requires_enrolment(env):
    _, api = env
    with pytest.raises(Exception):
        api.challenge("nobody", TX, Mode.PQ)


def test_device_identity_round_trip(enrolled, tmp_path):
    _, _, identity = enrolled
    identity.save(tmp_path / "dev")
    loaded = DeviceIdentity.load(tmp_path / "dev", identity.user_id, identity.mode)
    assert loaded.server_kem_pub == identity.server_kem_pub
    assert identity.suite.sig_public_bytes(loaded.sig_private) == identity.suite.sig_public_bytes(identity.sig_private)


def test_bundle_wire_format_round_trip(enrolled):
    _, api, identity = enrolled
    bundle = seal_evidence(identity, fresh(api, identity), synthetic_video(), synthetic_metadata())
    assert SealedBundle.from_bytes(bundle.to_bytes()) == bundle
