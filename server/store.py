"""Server key material and the SQLite store (enrolments, templates, signed verdicts, request log)."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from crypto import CryptoSuite, Mode


class ServerKeys:
    """Per-mode KEM and verdict-signing keys, plus a storage key and the action-derivation secret.

    Generated on first start into keys_dir. Verdict public keys are also written as
    verdict_<MODE>.pub so scripts/verify_verdict.py can check records offline.
    """

    def __init__(self, keys_dir: str | Path):
        self.dir = Path(keys_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.suites = {m: CryptoSuite(m) for m in Mode}
        self._kem = {m: self._load(f"kem_{m.value}.key", s.kem_keygen, s.load_kem_private, s) for m, s in self.suites.items()}
        self._sig = {m: self._load(f"verdict_{m.value}.key", s.sig_keygen, s.load_sig_private, s) for m, s in self.suites.items()}
        for m, s in self.suites.items():
            (self.dir / f"verdict_{m.value}.pub").write_bytes(s.sig_public_bytes(self._sig[m]))
        self.storage_key = self._secret("storage.key")
        self.action_secret = self._secret("actions.key")

    def _load(self, name, generate, load, suite: CryptoSuite):
        path = self.dir / name
        if path.exists():
            return load(path.read_bytes())
        key = generate()
        _write_private(path, suite.private_bytes(key))
        return key

    def _secret(self, name: str) -> bytes:
        path = self.dir / name
        if not path.exists():
            _write_private(path, os.urandom(32))
        return path.read_bytes()

    def kem_private(self, mode: Mode):
        return self._kem[Mode(mode)]

    def kem_public(self, mode: Mode) -> bytes:
        mode = Mode(mode)
        return self.suites[mode].kem_public_bytes(self._kem[mode])

    def verdict_private(self, mode: Mode):
        return self._sig[Mode(mode)]

    def verdict_public(self, mode: Mode) -> bytes:
        mode = Mode(mode)
        return self.suites[mode].sig_public_bytes(self._sig[mode])


def _write_private(path: Path, data: bytes) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as fh:
        fh.write(data)


SCHEMA = """
CREATE TABLE IF NOT EXISTS enrolments (
    user_id TEXT NOT NULL, mode TEXT NOT NULL, device_pub BLOB NOT NULL, created_ms INTEGER NOT NULL,
    PRIMARY KEY (user_id, mode));
CREATE TABLE IF NOT EXISTS templates (
    user_id TEXT PRIMARY KEY, iv BLOB NOT NULL, ct BLOB NOT NULL, model_version TEXT NOT NULL, created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS verdicts (
    challenge_id TEXT PRIMARY KEY, mode TEXT NOT NULL, record BLOB NOT NULL, signature BLOB NOT NULL, created_ms INTEGER NOT NULL);
CREATE TABLE IF NOT EXISTS requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT, challenge_id TEXT, mode TEXT, decision TEXT NOT NULL,
    reason_code TEXT, bytes_in INTEGER NOT NULL, timing_json TEXT NOT NULL, created_ms INTEGER NOT NULL);
"""


def now_ms() -> int:
    return time.time_ns() // 1_000_000


class Database:
    def __init__(self, path: str | Path, storage_key: bytes):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(SCHEMA)
        self._lock = threading.Lock()
        self._aead = AESGCM(storage_key)

    def _exec(self, sql: str, args: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, args)

    # enrolment ---------------------------------------------------------------

    def put_device_key(self, user_id: str, mode: Mode, device_pub: bytes) -> None:
        self._exec(
            "INSERT OR REPLACE INTO enrolments (user_id, mode, device_pub, created_ms) VALUES (?, ?, ?, ?)",
            (user_id, Mode(mode).value, device_pub, now_ms()),
        )

    def get_device_key(self, user_id: str, mode: Mode) -> bytes | None:
        row = self._exec("SELECT device_pub FROM enrolments WHERE user_id = ? AND mode = ?", (user_id, Mode(mode).value)).fetchone()
        return row[0] if row else None

    def put_template(self, user_id: str, embedding: np.ndarray, model_version: str) -> None:
        """Templates are encrypted at rest with the server storage key (AES-256-GCM, AAD = user_id|model)."""
        iv = os.urandom(12)
        ct = self._aead.encrypt(iv, np.asarray(embedding, np.float32).tobytes(), f"{user_id}|{model_version}".encode())
        self._exec(
            "INSERT OR REPLACE INTO templates (user_id, iv, ct, model_version, created_ms) VALUES (?, ?, ?, ?, ?)",
            (user_id, iv, ct, model_version, now_ms()),
        )

    def get_template(self, user_id: str) -> tuple[np.ndarray, str] | None:
        row = self._exec("SELECT iv, ct, model_version FROM templates WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        iv, ct, version = row
        plain = self._aead.decrypt(iv, ct, f"{user_id}|{version}".encode())
        return np.frombuffer(plain, np.float32).copy(), version

    # verdicts and request log ---------------------------------------------------

    def put_verdict(self, challenge_id: str, mode: Mode, record: bytes, signature: bytes) -> None:
        self._exec(
            "INSERT INTO verdicts (challenge_id, mode, record, signature, created_ms) VALUES (?, ?, ?, ?, ?)",
            (challenge_id, Mode(mode).value, record, signature, now_ms()),
        )

    def get_verdict(self, challenge_id: str) -> tuple[str, bytes, bytes] | None:
        row = self._exec("SELECT mode, record, signature FROM verdicts WHERE challenge_id = ?", (challenge_id,)).fetchone()
        return (row[0], row[1], row[2]) if row else None

    def add_request(
        self, challenge_id: str | None, mode: str | None, decision: str, reason: str | None, bytes_in: int, timing: dict
    ) -> int:
        cur = self._exec(
            "INSERT INTO requests (challenge_id, mode, decision, reason_code, bytes_in, timing_json, created_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (challenge_id, mode, decision, reason, bytes_in, json.dumps(timing, sort_keys=True), now_ms()),
        )
        return int(cur.lastrowid)

    def get_request(self, request_id: int) -> dict | None:
        cur = self._exec(
            "SELECT request_id, challenge_id, mode, decision, reason_code, bytes_in, timing_json, created_ms"
            " FROM requests WHERE request_id = ?",
            (request_id,),
        )
        row = cur.fetchone()
        if not row:
            return None
        keys = [d[0] for d in cur.description]
        out = dict(zip(keys, row))
        out["timing"] = json.loads(out.pop("timing_json"))
        return out

    def close(self) -> None:
        self._conn.close()
