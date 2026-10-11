"""Protocol checks for POST /verify, in a fixed order. Stop at the first failure.

| # | Check                                                     | Reason code        |
|---|-----------------------------------------------------------|--------------------|
| 1 | Challenge exists, unexpired, unused -> consumed (GETDEL)  | EXPIRED / REPLAY   |
| 2 | Bundle mode matches the mode stored with the challenge    | MODE_MISMATCH      |
| 3 | Decapsulate, derive K, decrypt (GCM tag valid)            | TAMPER             |
| 4 | Manifest nonce, tx_hash, challenge_id, mode match         | TX_MISMATCH        |
| 5 | Recomputed SHA-384(video) matches manifest.sample_hash    | HASH_MISMATCH      |
| 6 | Signature verifies under the enrolled device key          | SIG_INVALID        |

The challenge lookup runs before the mode comparison because the mode to compare
against is stored with the challenge, and so a downgraded request still burns it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from crypto import CryptoSuite, Evidence, Manifest, Mode, SealedBundle, TamperError, sha384_hex
from server.challenge import Challenge, ChallengeStore
from server.store import Database, ServerKeys
from server.timing import Timer


class Reason(str, Enum):
    EXPIRED = "EXPIRED"
    REPLAY = "REPLAY"
    MODE_MISMATCH = "MODE_MISMATCH"
    TAMPER = "TAMPER"
    TX_MISMATCH = "TX_MISMATCH"
    HASH_MISMATCH = "HASH_MISMATCH"
    SIG_INVALID = "SIG_INVALID"
    NO_MATCH = "NO_MATCH"
    PAD_FAIL = "PAD_FAIL"
    MORPH_FLAG = "MORPH_FLAG"
    DEEPFAKE_FAIL = "DEEPFAKE_FAIL"
    SCRIPT_FAIL = "SCRIPT_FAIL"
    INJECTION_FAIL = "INJECTION_FAIL"


PROTOCOL_REASONS = frozenset(
    {Reason.EXPIRED, Reason.REPLAY, Reason.MODE_MISMATCH, Reason.TAMPER, Reason.TX_MISMATCH, Reason.HASH_MISMATCH, Reason.SIG_INVALID}
)


@dataclass
class CheckResult:
    reason: Reason | None = None
    bundle: SealedBundle | None = None
    challenge: Challenge | None = None  # set once the challenge has been consumed by this request
    evidence: Evidence | None = None
    manifest: Manifest | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


def run_checks(body: bytes, store: ChallengeStore, keys: ServerKeys, db: Database, timer: Timer) -> CheckResult:
    res = CheckResult()
    with timer.stage("parse"):
        try:
            res.bundle = SealedBundle.from_bytes(body)
        except (ValueError, UnicodeDecodeError):
            res.reason = Reason.TAMPER
            return res
    bundle = res.bundle

    with timer.stage("check1_challenge"):
        challenge, failure = store.consume(bundle.challenge_id)
    if challenge is None:
        res.reason = Reason(failure)
        return res
    res.challenge = challenge

    if bundle.mode.value != challenge.mode:  # check 2
        res.reason = Reason.MODE_MISMATCH
        return res
    suite = CryptoSuite(Mode(challenge.mode))

    with timer.stage("check3_unseal"):
        try:
            res.evidence = suite.unseal(keys.kem_private(suite.mode), bundle, challenge.nonce)
        except TamperError:
            res.reason = Reason.TAMPER
            return res
    evidence = res.evidence

    with timer.stage("check4_binding"):
        try:
            res.manifest = manifest = Manifest.from_bytes(evidence.manifest)
        except (ValueError, TypeError, UnicodeDecodeError):
            res.reason = Reason.TX_MISMATCH
            return res
        bound = (
            manifest.nonce == challenge.nonce.hex()
            and manifest.tx_hash == challenge.tx_hash
            and manifest.challenge_id == challenge.challenge_id
            and manifest.mode == challenge.mode
        )
    if not bound:
        res.reason = Reason.TX_MISMATCH
        return res

    with timer.stage("check5_hash"):
        hash_ok = sha384_hex(evidence.video) == manifest.sample_hash
    if not hash_ok:
        res.reason = Reason.HASH_MISMATCH
        return res

    with timer.stage("check6_signature"):
        device_pub = db.get_device_key(challenge.user_id, suite.mode)
        sig_ok = device_pub is not None and suite.verify(device_pub, evidence.signature, manifest.digest())
    if not sig_ok:
        res.reason = Reason.SIG_INVALID
    return res
