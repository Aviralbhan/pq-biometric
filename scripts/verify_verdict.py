"""Re-check signed verdict records offline: anyone with the server's verdict public key can do this.

    python scripts/verify_verdict.py --challenge-id <id>
    python scripts/verify_verdict.py --all
    python scripts/verify_verdict.py --record record.json --signature sig.bin --mode PQ --public-key var/server_keys/verdict_PQ.pub

Checks the ML-DSA-65 (PQ) or Ed25519 (CLASSICAL) signature over SHA-384(record) and
that the record's signer id matches the key. Exit code 1 if any record fails.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from crypto import Mode  # noqa: E402
from server.verdict import signer_id, verify_record  # noqa: E402
from settings import load_config, resolve  # noqa: E402


def check(mode: str, record: bytes, signature: bytes, keys_dir: Path, public_key: bytes | None = None) -> tuple[bool, str]:
    pub = public_key or (keys_dir / f"verdict_{mode}.pub").read_bytes()
    if not verify_record(Mode(mode), pub, record, signature):
        return False, "signature does not verify"
    rec = json.loads(record)
    if rec.get("signer") != signer_id(pub):
        return False, "signer id does not match the key"
    if rec.get("mode") != mode:
        return False, "record mode differs from stored mode"
    return True, f"{rec['decision']} ({rec.get('reason_code') or 'no failure'})"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--challenge-id")
    g.add_argument("--all", action="store_true")
    g.add_argument("--record")
    ap.add_argument("--signature")
    ap.add_argument("--mode", choices=[m.value for m in Mode])
    ap.add_argument("--public-key")
    ap.add_argument("--db")
    ap.add_argument("--keys-dir")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    keys_dir = Path(args.keys_dir) if args.keys_dir else resolve(config["server"]["keys_dir"])

    if args.record:
        if not (args.signature and args.mode):
            ap.error("--record needs --signature and --mode")
        pub = Path(args.public_key).read_bytes() if args.public_key else None
        ok, msg = check(args.mode, Path(args.record).read_bytes(), Path(args.signature).read_bytes(), keys_dir, pub)
        print(f"{'OK  ' if ok else 'FAIL'} {args.record}: {msg}")
        return 0 if ok else 1

    db = Path(args.db) if args.db else resolve(config["server"]["db_path"])
    with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
        if args.all:
            rows = conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts ORDER BY created_ms").fetchall()
        else:
            rows = conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts WHERE challenge_id = ?", (args.challenge_id,)).fetchall()
    if not rows:
        print("no matching verdict", file=sys.stderr)
        return 1
    failures = 0
    for cid, mode, record, signature in rows:
        ok, msg = check(mode, record, signature, keys_dir)
        failures += not ok
        if not ok or not args.all:
            print(f"{'OK  ' if ok else 'FAIL'} {cid} [{mode}]: {msg}")
    if args.all:
        print(f"{len(rows) - failures}/{len(rows)} verdicts verify")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
