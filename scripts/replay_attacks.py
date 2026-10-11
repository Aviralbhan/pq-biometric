"""S9 replay suite: protocol attacks against the binding and freshness checks, plus genuine runs.

    python scripts/replay_attacks.py --runs 1000 --modes PQ CLASSICAL [--session var/sessions/alice1] [--url http://127.0.0.1:8000]

Attacks (each uses its own fresh challenge; expected reason code in brackets):
  exact_resubmit  a captured bundle sent a second time                         [REPLAY]
  expired         a valid bundle sent after the challenge TTL                  [EXPIRED]
  old_nonce       old evidence (old manifest) re-sealed under a new challenge  [TX_MISMATCH]
  swapped_tx      manifest re-bound to a different transaction                 [TX_MISMATCH]
  flipped_byte    one random bit flipped in kem_ct / iv / payload              [TAMPER]
old_nonce and swapped_tx model a compromised client that holds the device key; the
others need no key. Genuine runs must never fail a protocol check (false rejection).

Writes eval/results/replay.csv.
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from client.protocol import check_challenge, make_manifest, seal_evidence  # noqa: E402
from crypto import Mode  # noqa: E402
from eval.harness import RESULTS, connect, session_or_synthetic, write_csv  # noqa: E402
from server.checks import PROTOCOL_REASONS  # noqa: E402
from settings import load_config  # noqa: E402

TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}
TX_OTHER = {"type": "payment", "amount": 50000, "currency": "INR", "payee": "attacker"}
EXPECTED = {
    "exact_resubmit": "REPLAY",
    "expired": "EXPIRED",
    "old_nonce": "TX_MISMATCH",
    "swapped_tx": "TX_MISMATCH",
    "flipped_byte": "TAMPER",
}
PROTOCOL_CODES = {r.value for r in PROTOCOL_REASONS}


def flip_random_bit(bundle, rng: random.Random):
    field = rng.choice(["kem_ct", "iv", "payload"])
    data = bytearray(getattr(bundle, field))
    i = rng.randrange(len(data))
    data[i] ^= 1 << rng.randrange(8)
    return replace(bundle, **{field: bytes(data)})


def run(conn, identity, video, meta, runs: int, ttl: int, rng: random.Random, log=print) -> list[dict]:
    api = conn.api
    user, mode = identity.user_id, identity.mode

    def challenge(tx=TX):
        ch = api.challenge(user, tx, mode)
        check_challenge(identity, ch, tx)
        return ch

    rows = []
    outcomes: dict[str, list[str | None]] = {k: [] for k in ["genuine", *EXPECTED]}

    t0 = time.perf_counter()
    for _ in range(runs):
        resp, rid = api.verify(seal_evidence(identity, challenge(), video, meta))
        outcomes["genuine"].append(conn.reason(rid))
        bundle = seal_evidence(identity, challenge(), video, meta)
        api.verify(bundle)
        outcomes["exact_resubmit"].append(conn.reason(api.verify(bundle)[1]))

        old = challenge()
        new = challenge()
        outcomes["old_nonce"].append(conn.reason(api.verify(seal_evidence(identity, new, video, meta, make_manifest(identity, old, video)))[1]))

        ch_a, ch_b = challenge(TX), challenge(TX_OTHER)
        rebound = replace(make_manifest(identity, ch_b, video), tx_hash=ch_a["tx_hash"])
        outcomes["swapped_tx"].append(conn.reason(api.verify(seal_evidence(identity, ch_b, video, meta, rebound))[1]))

        tampered = flip_random_bit(seal_evidence(identity, challenge(), video, meta), rng)
        outcomes["flipped_byte"].append(conn.reason(api.verify(tampered)[1]))
    log(f"  {mode.value}: genuine + 4 attacks x {runs} in {time.perf_counter() - t0:.0f} s")

    # Expired: issue every challenge first, wait out the TTL once, then submit.
    pending = [challenge() for _ in range(runs)]
    log(f"  {mode.value}: waiting {ttl + 2} s for {runs} challenges to expire")
    time.sleep(ttl + 2)
    for ch in pending:
        outcomes["expired"].append(conn.reason(api.verify(seal_evidence(identity, ch, video, meta))[1]))

    for attack, expected in EXPECTED.items():
        got = outcomes[attack]
        rows.append({
            "mode": mode.value,
            "case": attack,
            "runs": len(got),
            "expected_code": expected,
            "rejected": sum(r is not None for r in got),
            "correct_code": sum(r == expected for r in got),
            "rejection_rate": sum(r is not None for r in got) / len(got),
            "correct_code_rate": sum(r == expected for r in got) / len(got),
            "other_codes": ";".join(sorted({str(r) for r in got if r != expected})),
        })
    g = outcomes["genuine"]
    protocol_fr = sum(r in PROTOCOL_CODES for r in g)
    rows.append({
        "mode": mode.value,
        "case": "genuine",
        "runs": len(g),
        "expected_code": "",
        "rejected": sum(r is not None for r in g),
        "correct_code": len(g) - protocol_fr,
        "rejection_rate": sum(r is not None for r in g) / len(g),
        "correct_code_rate": 1 - protocol_fr / len(g),
        "other_codes": ";".join(sorted({str(r) for r in g if r is not None})),
        "protocol_false_rejections": protocol_fr,
    })
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--modes", nargs="+", default=[m.value for m in Mode], choices=[m.value for m in Mode])
    ap.add_argument("--session", help="recorded session dir used as the genuine video (default: synthetic)")
    ap.add_argument("--reference", help="reference JPEG for enrolment (default: a frame of the session)")
    ap.add_argument("--url")
    ap.add_argument("--config")
    ap.add_argument("--out", default=str(RESULTS / "replay.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    config = load_config(args.config)
    video, meta, ref, source = session_or_synthetic(args.session, args.reference)
    rng = random.Random(args.seed)
    rows = []
    with connect(config, args.url) as conn:
        for m in args.modes:
            identity = conn.api.enroll(f"replay-{m.lower()}", m, ref)
            rows += run(conn, identity, video, meta, args.runs, int(config["challenge"]["ttl_seconds"]), rng)
    for r in rows:
        r["video"] = source
    path = write_csv(args.out, rows)
    print(f"wrote {path}")
    for r in rows:
        print(f"  {r['mode']:9} {r['case']:15} rejected {r['rejected']}/{r['runs']}  correct code {r['correct_code']}/{r['runs']}")
    attacks_ok = all(r["correct_code_rate"] == 1.0 for r in rows if r["case"] != "genuine")
    genuine_ok = all(r.get("protocol_false_rejections", 0) == 0 for r in rows if r["case"] == "genuine")
    return 0 if attacks_ok and genuine_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
