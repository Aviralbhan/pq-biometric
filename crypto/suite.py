"""Crypto suite for Layer B, in CLASSICAL or PQ mode behind one interface.

| Mode      | Key exchange                         | Signature | Symmetric                    |
|-----------|--------------------------------------|-----------|------------------------------|
| CLASSICAL | X25519 (ephemeral client, static srv) | Ed25519   | HKDF-SHA384 -> AES-256-GCM   |
| PQ        | ML-KEM-768                           | ML-DSA-65 | HKDF-SHA384 -> AES-256-GCM   |

X25519 is wrapped as a KEM (the client's ephemeral public key is the "ciphertext")
so client and server code is identical in both modes. Only the keys differ.

Layer B (client):
    sig = sign(device_key, SHA-384(manifest))
    (kem_ct, ss) = encapsulate(server_kem_pub)
    K = HKDF-SHA384(ikm=ss, salt=nonce, info="BEA-v1|<mode>|<challenge_id>")
    payload = AES-256-GCM(K, video || metadata || manifest || sig, AAD="<mode>|<challenge_id>")
    send {challenge_id, mode, kem_ct, iv, payload}
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from enum import Enum
from typing import Union

from cryptography.exceptions import InvalidSignature, InvalidTag
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ed25519, mldsa, mlkem, x25519
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

PROTOCOL_LABEL = "BEA-v1"
NONCE_LEN = 32
IV_LEN = 12
KEY_LEN = 32
_LEN_PREFIX = struct.Struct(">I")

KemPrivateKey = Union[x25519.X25519PrivateKey, mlkem.MLKEM768PrivateKey]
SigPrivateKey = Union[ed25519.Ed25519PrivateKey, mldsa.MLDSA65PrivateKey]


class Mode(str, Enum):
    CLASSICAL = "CLASSICAL"
    PQ = "PQ"


class TamperError(Exception):
    """Decapsulation, key derivation or AES-GCM authentication failed (reason code TAMPER)."""


@dataclass(frozen=True)
class Sizes:
    kem_pub: int
    kem_ct: int
    sig_pub: int
    sig: int


SIZES = {
    Mode.CLASSICAL: Sizes(kem_pub=32, kem_ct=32, sig_pub=32, sig=64),
    Mode.PQ: Sizes(kem_pub=1184, kem_ct=1088, sig_pub=1952, sig=3309),
}


@dataclass(frozen=True)
class Evidence:
    """Plaintext carried inside the encrypted payload."""

    video: bytes
    metadata: bytes  # canonical JSON of capture metadata
    manifest: bytes  # canonical JSON of the Manifest
    signature: bytes  # device signature over SHA-384(manifest)


@dataclass(frozen=True)
class SealedBundle:
    """What the client sends to POST /verify. challenge_id and mode travel in the clear as AAD."""

    challenge_id: str
    mode: Mode
    kem_ct: bytes
    iv: bytes
    payload: bytes

    def wire_size(self) -> int:
        return len(self.challenge_id.encode()) + len(self.mode.value) + len(self.kem_ct) + len(self.iv) + len(self.payload)

    def to_bytes(self) -> bytes:
        """HTTP body for POST /verify: length-prefixed challenge_id, mode, kem_ct, iv, payload."""
        return _pack(self.challenge_id.encode(), self.mode.value.encode(), self.kem_ct, self.iv, self.payload)

    @classmethod
    def from_bytes(cls, data: bytes) -> SealedBundle:
        challenge_id, mode, kem_ct, iv, payload = _unpack(data, 5)
        return cls(challenge_id=challenge_id.decode("ascii"), mode=Mode(mode.decode("ascii")), kem_ct=kem_ct, iv=iv, payload=payload)


def _pack(*parts: bytes) -> bytes:
    return b"".join(_LEN_PREFIX.pack(len(p)) + p for p in parts)


def _unpack(data: bytes, count: int) -> list[bytes]:
    parts: list[bytes] = []
    offset = 0
    for _ in range(count):
        if offset + _LEN_PREFIX.size > len(data):
            raise ValueError("truncated payload")
        (n,) = _LEN_PREFIX.unpack_from(data, offset)
        offset += _LEN_PREFIX.size
        if offset + n > len(data):
            raise ValueError("truncated payload")
        parts.append(data[offset : offset + n])
        offset += n
    if offset != len(data):
        raise ValueError("trailing bytes in payload")
    return parts


class CryptoSuite:
    def __init__(self, mode: Mode | str) -> None:
        self.mode = Mode(mode)
        self.sizes = SIZES[self.mode]
        if self.mode is Mode.PQ:
            self._kem_priv_type, self._sig_priv_type = mlkem.MLKEM768PrivateKey, mldsa.MLDSA65PrivateKey
        else:
            self._kem_priv_type, self._sig_priv_type = x25519.X25519PrivateKey, ed25519.Ed25519PrivateKey

    # ---- key encapsulation -------------------------------------------------

    def kem_keygen(self) -> KemPrivateKey:
        return self._kem_priv_type.generate()

    def kem_public_bytes(self, priv: KemPrivateKey) -> bytes:
        self._check(priv, self._kem_priv_type)
        return priv.public_key().public_bytes_raw()

    def encapsulate(self, server_kem_pub: bytes) -> tuple[bytes, bytes]:
        """Return (kem_ct, shared_secret)."""
        if self.mode is Mode.PQ:
            ss, ct = mlkem.MLKEM768PublicKey.from_public_bytes(server_kem_pub).encapsulate()
            return ct, ss
        eph = x25519.X25519PrivateKey.generate()
        ss = eph.exchange(x25519.X25519PublicKey.from_public_bytes(server_kem_pub))
        return eph.public_key().public_bytes_raw(), ss

    def decapsulate(self, server_kem_priv: KemPrivateKey, kem_ct: bytes) -> bytes:
        self._check(server_kem_priv, self._kem_priv_type)
        if self.mode is Mode.PQ:
            return server_kem_priv.decapsulate(kem_ct)
        return server_kem_priv.exchange(x25519.X25519PublicKey.from_public_bytes(kem_ct))

    # ---- signatures --------------------------------------------------------

    def sig_keygen(self) -> SigPrivateKey:
        return self._sig_priv_type.generate()

    def sig_public_bytes(self, priv: SigPrivateKey) -> bytes:
        self._check(priv, self._sig_priv_type)
        return priv.public_key().public_bytes_raw()

    def sign(self, priv: SigPrivateKey, data: bytes) -> bytes:
        self._check(priv, self._sig_priv_type)
        return priv.sign(data)

    def verify(self, sig_pub: bytes, signature: bytes, data: bytes) -> bool:
        try:
            if self.mode is Mode.PQ:
                mldsa.MLDSA65PublicKey.from_public_bytes(sig_pub).verify(signature, data)
            else:
                ed25519.Ed25519PublicKey.from_public_bytes(sig_pub).verify(signature, data)
        except (InvalidSignature, ValueError):
            return False
        return True

    # ---- key storage -------------------------------------------------------

    def private_bytes(self, priv: KemPrivateKey | SigPrivateKey) -> bytes:
        """Raw private key (X25519/Ed25519) or seed (ML-KEM 64 B, ML-DSA 32 B)."""
        self._check(priv, (self._kem_priv_type, self._sig_priv_type))
        return priv.private_bytes_raw()

    def load_kem_private(self, raw: bytes) -> KemPrivateKey:
        if self.mode is Mode.PQ:
            return mlkem.MLKEM768PrivateKey.from_seed_bytes(raw)
        return x25519.X25519PrivateKey.from_private_bytes(raw)

    def load_sig_private(self, raw: bytes) -> SigPrivateKey:
        if self.mode is Mode.PQ:
            return mldsa.MLDSA65PrivateKey.from_seed_bytes(raw)
        return ed25519.Ed25519PrivateKey.from_private_bytes(raw)

    # ---- symmetric layer ---------------------------------------------------

    def derive_key(self, shared_secret: bytes, nonce: bytes, challenge_id: str) -> bytes:
        info = f"{PROTOCOL_LABEL}|{self.mode.value}|{challenge_id}".encode()
        return HKDF(algorithm=hashes.SHA384(), length=KEY_LEN, salt=nonce, info=info).derive(shared_secret)

    def aad(self, challenge_id: str) -> bytes:
        return f"{self.mode.value}|{challenge_id}".encode()

    @staticmethod
    def encrypt(key: bytes, plaintext: bytes, aad: bytes) -> tuple[bytes, bytes]:
        iv = os.urandom(IV_LEN)
        return iv, AESGCM(key).encrypt(iv, plaintext, aad)

    @staticmethod
    def decrypt(key: bytes, iv: bytes, ciphertext: bytes, aad: bytes) -> bytes:
        try:
            return AESGCM(key).decrypt(iv, ciphertext, aad)
        except (InvalidTag, ValueError) as exc:
            raise TamperError("AES-GCM authentication failed") from exc

    # ---- Layer B -----------------------------------------------------------

    def seal(self, server_kem_pub: bytes, challenge_id: str, nonce: bytes, evidence: Evidence) -> SealedBundle:
        if len(nonce) != NONCE_LEN:
            raise ValueError(f"nonce must be {NONCE_LEN} bytes")
        kem_ct, ss = self.encapsulate(server_kem_pub)
        key = self.derive_key(ss, nonce, challenge_id)
        plaintext = _pack(evidence.video, evidence.metadata, evidence.manifest, evidence.signature)
        iv, payload = self.encrypt(key, plaintext, self.aad(challenge_id))
        return SealedBundle(challenge_id=challenge_id, mode=self.mode, kem_ct=kem_ct, iv=iv, payload=payload)

    def unseal(self, server_kem_priv: KemPrivateKey, bundle: SealedBundle, nonce: bytes) -> Evidence:
        """Check 3: decapsulate, derive K, decrypt. Raises TamperError on any failure.

        The caller must already have matched bundle.mode against the stored challenge (check 1).
        """
        if bundle.mode is not self.mode:
            raise ValueError(f"bundle mode {bundle.mode.value} does not match suite mode {self.mode.value}")
        try:
            ss = self.decapsulate(server_kem_priv, bundle.kem_ct)
        except ValueError as exc:
            raise TamperError("key decapsulation failed") from exc
        key = self.derive_key(ss, nonce, bundle.challenge_id)
        plaintext = self.decrypt(key, bundle.iv, bundle.payload, self.aad(bundle.challenge_id))
        try:
            video, metadata, manifest, signature = _unpack(plaintext, 4)
        except ValueError as exc:
            raise TamperError("malformed payload") from exc
        return Evidence(video=video, metadata=metadata, manifest=manifest, signature=signature)

    # ---- helpers -----------------------------------------------------------

    def _check(self, key: object, expected: type | tuple[type, ...]) -> None:
        if not isinstance(key, expected):
            raise TypeError(f"{type(key).__name__} is not a {self.mode.value} mode key")
