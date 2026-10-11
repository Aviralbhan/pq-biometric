"""Challenge issue and single-use consumption (Redis).

A challenge lives at chal:<id> for ttl_seconds. Consuming it uses GETDEL, so two
concurrent submissions can never both succeed. A consumed id is remembered at
chal:used:<id> so a re-submission is reported as REPLAY rather than EXPIRED.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import uuid
from dataclasses import asdict, dataclass
from typing import Any

from crypto import Mode, tx_hash
from server.store import now_ms

CHALLENGE_ID_RE = re.compile(r"^[0-9a-f]{32}$")


@dataclass(frozen=True)
class Challenge:
    challenge_id: str
    user_id: str
    mode: str
    nonce: bytes
    tx_hash: str
    actions: list[str]
    issued_ms: int
    expires_ms: int

    def to_json(self) -> str:
        d = asdict(self)
        d["nonce"] = self.nonce.hex()
        return json.dumps(d, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> Challenge:
        d = json.loads(raw)
        d["nonce"] = bytes.fromhex(d["nonce"])
        return cls(**d)


def derive_actions(secret: bytes, nonce: bytes, pool: list[str], n: int) -> list[str]:
    """n distinct actions in an order fixed by HMAC-SHA256(secret, nonce) (Fisher-Yates on the MAC bytes)."""
    if n > len(pool):
        raise ValueError("more actions requested than available")
    mac = hmac.new(secret, nonce, hashlib.sha256).digest()
    items = list(pool)
    for i in range(len(items) - 1, 0, -1):
        j = mac[len(items) - 1 - i] % (i + 1)
        items[i], items[j] = items[j], items[i]
    return items[:n]


def make_redis(url: str):
    if url.startswith("fakeredis://"):
        import fakeredis

        return fakeredis.FakeRedis()
    import redis

    return redis.Redis.from_url(url)


class ChallengeStore:
    def __init__(self, redis_client: Any, config: dict[str, Any], action_secret: bytes):
        c = config["challenge"]
        self.r = redis_client
        self.ttl = int(c["ttl_seconds"])
        self.used_ttl = int(c["used_marker_ttl_seconds"])
        self.nonce_bytes = int(c["nonce_bytes"])
        self.pool = list(c["actions"])
        self.n_actions = int(c["num_actions"])
        self.secret = action_secret

    def issue(self, user_id: str, mode: Mode, tx_context: dict[str, Any]) -> Challenge:
        nonce = os.urandom(self.nonce_bytes)
        issued = now_ms()
        ch = Challenge(
            challenge_id=uuid.uuid4().hex,
            user_id=user_id,
            mode=Mode(mode).value,
            nonce=nonce,
            tx_hash=tx_hash(tx_context),
            actions=derive_actions(self.secret, nonce, self.pool, self.n_actions),
            issued_ms=issued,
            expires_ms=issued + self.ttl * 1000,
        )
        self.r.set(f"chal:{ch.challenge_id}", ch.to_json(), ex=self.ttl)
        return ch

    def consume(self, challenge_id: str) -> tuple[Challenge | None, str | None]:
        """Return (challenge, None) exactly once; afterwards (None, "REPLAY"). Unknown or timed out: "EXPIRED"."""
        if not CHALLENGE_ID_RE.match(challenge_id):
            return None, "EXPIRED"
        raw = self.r.getdel(f"chal:{challenge_id}")
        if raw is None:
            # A concurrent duplicate can land between the GETDEL above and the marker below; it is
            # still rejected, only labelled EXPIRED instead of REPLAY.
            return None, ("REPLAY" if self.r.exists(f"chal:used:{challenge_id}") else "EXPIRED")
        self.r.set(f"chal:used:{challenge_id}", 1, ex=self.used_ttl)
        ch = Challenge.from_json(raw)
        if now_ms() > ch.expires_ms:
            return None, "EXPIRED"
        return ch, None
