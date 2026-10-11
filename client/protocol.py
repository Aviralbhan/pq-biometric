"""Client side of the protocol, independent of the camera.

Used by the capture CLI, the replay-attack suite, the performance harness and tests.
The server's KEM key is pinned at enrolment; a challenge that names a different key
is refused, so a man in the middle cannot substitute its own.
"""

from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from crypto import CryptoSuite, Evidence, Manifest, Mode, SealedBundle, canonical_json, sha384_hex, tx_hash


class ProtocolError(Exception):
    """The challenge does not match what this client expects; nothing is sent."""


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    return base64.b64decode(text, validate=True)


@dataclass
class DeviceIdentity:
    user_id: str
    mode: Mode
    sig_private: Any
    server_kem_pub: bytes  # pinned at enrolment
    verdict_pub: bytes

    @property
    def suite(self) -> CryptoSuite:
        return CryptoSuite(self.mode)

    def save(self, device_dir: str | Path) -> Path:
        d = Path(device_dir) / self.user_id / self.mode.value
        d.mkdir(parents=True, exist_ok=True)
        (d / "device.key").write_bytes(self.suite.private_bytes(self.sig_private))
        (d / "server_kem.pub").write_bytes(self.server_kem_pub)
        (d / "verdict.pub").write_bytes(self.verdict_pub)
        return d

    @classmethod
    def load(cls, device_dir: str | Path, user_id: str, mode: Mode | str) -> DeviceIdentity:
        mode = Mode(mode)
        d = Path(device_dir) / user_id / mode.value
        if not (d / "device.key").exists():
            raise FileNotFoundError(f"no enrolment for {user_id}/{mode.value} in {device_dir}; run client.enroll first")
        suite = CryptoSuite(mode)
        return cls(
            user_id=user_id,
            mode=mode,
            sig_private=suite.load_sig_private((d / "device.key").read_bytes()),
            server_kem_pub=(d / "server_kem.pub").read_bytes(),
            verdict_pub=(d / "verdict.pub").read_bytes(),
        )


def make_manifest(identity: DeviceIdentity, challenge: dict[str, Any], video: bytes, timestamp_ms: int | None = None) -> Manifest:
    return Manifest(
        mode=challenge["mode"],
        challenge_id=challenge["challenge_id"],
        nonce=b64d(challenge["nonce"]).hex(),
        tx_hash=challenge["tx_hash"],
        sample_hash=sha384_hex(video),
        timestamp=timestamp_ms if timestamp_ms is not None else time.time_ns() // 1_000_000,
    )


def check_challenge(identity: DeviceIdentity, challenge: dict[str, Any], tx_context: dict[str, Any]) -> None:
    if Mode(challenge["mode"]) is not identity.mode:
        raise ProtocolError(f"challenge mode {challenge['mode']} differs from enrolled mode {identity.mode.value}")
    if b64d(challenge["server_kem_pub"]) != identity.server_kem_pub:
        raise ProtocolError("server KEM key differs from the key pinned at enrolment")
    if tx_hash(tx_context) != challenge["tx_hash"]:
        raise ProtocolError("challenge is bound to a different transaction than the one shown")


def seal_evidence(
    identity: DeviceIdentity,
    challenge: dict[str, Any],
    video: bytes,
    metadata: dict[str, Any],
    manifest: Manifest | None = None,
) -> SealedBundle:
    """Sign SHA-384(manifest) with the device key, then seal everything to the pinned server key."""
    suite = identity.suite
    manifest = manifest or make_manifest(identity, challenge, video)
    manifest_bytes = manifest.to_bytes()
    evidence = Evidence(
        video=video,
        metadata=canonical_json(metadata),
        manifest=manifest_bytes,
        signature=suite.sign(identity.sig_private, manifest.digest()),
    )
    return suite.seal(identity.server_kem_pub, challenge["challenge_id"], b64d(challenge["nonce"]), evidence)


class ApiClient:
    """Thin HTTP wrapper. Accepts any httpx.Client, including FastAPI's TestClient."""

    def __init__(self, http: httpx.Client | str, timeout: float = 120.0):
        self.http = httpx.Client(base_url=http, timeout=timeout) if isinstance(http, str) else http

    def enroll(self, user_id: str, mode: Mode | str, reference_jpeg: bytes, sig_private: Any | None = None) -> DeviceIdentity:
        mode = Mode(mode)
        suite = CryptoSuite(mode)
        sig_private = sig_private or suite.sig_keygen()
        r = self.http.post(
            "/enroll",
            json={
                "user_id": user_id,
                "mode": mode.value,
                "device_pub": b64e(suite.sig_public_bytes(sig_private)),
                "reference_jpeg": b64e(reference_jpeg),
            },
        )
        r.raise_for_status()
        body = r.json()
        return DeviceIdentity(user_id, mode, sig_private, b64d(body["server_kem_pub"]), b64d(body["verdict_pub"]))

    def challenge(self, user_id: str, tx_context: dict[str, Any], mode: Mode | str | None = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"user_id": user_id, "tx_context": tx_context}
        if mode is not None:
            payload["mode"] = Mode(mode).value
        r = self.http.post("/challenge", json=payload)
        r.raise_for_status()
        return r.json()

    def verify_bytes(self, body: bytes) -> tuple[dict[str, Any], int | None]:
        r = self.http.post("/verify", content=body, headers={"Content-Type": "application/octet-stream"})
        r.raise_for_status()
        rid = r.headers.get("X-Request-Id")
        return r.json(), int(rid) if rid else None

    def verify(self, bundle: SealedBundle) -> tuple[dict[str, Any], int | None]:
        return self.verify_bytes(bundle.to_bytes())

    def authenticate(
        self, identity: DeviceIdentity, tx_context: dict[str, Any], video: bytes, metadata: dict[str, Any]
    ) -> tuple[dict[str, Any], int | None]:
        """Challenge -> check -> seal -> verify, for an already captured video."""
        ch = self.challenge(identity.user_id, tx_context, identity.mode)
        check_challenge(identity, ch, tx_context)
        return self.verify(seal_evidence(identity, ch, video, metadata))
