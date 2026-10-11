"""Evidence manifest: canonical JSON encoding and SHA-384 hashing.

The manifest is what the device key signs. Both client and server must produce
byte-identical encodings, so the encoding is fixed: UTF-8 JSON with sorted keys
and no whitespace. Byte values (nonce, hashes) are carried as lowercase hex.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

MANIFEST_FIELDS = ("challenge_id", "mode", "nonce", "sample_hash", "timestamp", "tx_hash")


def canonical_json(obj: Any) -> bytes:
    """Deterministic JSON: sorted keys, no spaces, UTF-8. Rejects NaN/Infinity."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def sha384(data: bytes) -> bytes:
    return hashlib.sha384(data).digest()


def sha384_hex(data: bytes) -> str:
    return hashlib.sha384(data).hexdigest()


def tx_hash(tx_context: dict[str, Any]) -> str:
    """SHA-384 of the canonical transaction context, as hex."""
    return sha384_hex(canonical_json(tx_context))


@dataclass(frozen=True)
class Manifest:
    mode: str
    challenge_id: str
    nonce: str  # hex of the 32-byte challenge nonce
    tx_hash: str  # hex SHA-384 of canonical tx_context
    sample_hash: str  # hex SHA-384 of the raw video bytes
    timestamp: int  # client capture time, Unix milliseconds

    def to_bytes(self) -> bytes:
        return canonical_json(asdict(self))

    def digest(self) -> bytes:
        """SHA-384(manifest): the value the device key signs."""
        return sha384(self.to_bytes())

    @classmethod
    def from_bytes(cls, data: bytes) -> Manifest:
        obj = json.loads(data.decode("utf-8"))
        if not isinstance(obj, dict) or tuple(sorted(obj)) != MANIFEST_FIELDS:
            raise ValueError("manifest has missing or unexpected fields")
        manifest = cls(**obj)
        if manifest.to_bytes() != data:
            raise ValueError("manifest is not in canonical form")
        return manifest
