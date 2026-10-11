"""Crypto micro-benchmark: every Layer B and verdict operation, 1,000 runs per op per mode.

    python -m eval.crypto_bench [--runs 1000] [--payload-kb 2048]

Writes eval/results/crypto_bench.csv (timings) and eval/results/crypto_sizes.csv (bytes).
Timings use time.perf_counter_ns() around single calls, after 20 warm-up calls.
"""

from __future__ import annotations

import argparse
import os
import platform
import time
from collections.abc import Callable

import cryptography
import numpy as np

from crypto import CryptoSuite, Evidence, Mode, sha384
from eval.harness import RESULTS, write_csv
from server.verdict import build_record, sign_record


def bench(fn: Callable[[], object], runs: int, warmup: int = 20) -> np.ndarray:
    for _ in range(warmup):
        fn()
    out = np.empty(runs)
    for i in range(runs):
        t = time.perf_counter_ns()
        fn()
        out[i] = (time.perf_counter_ns() - t) / 1000.0
    return out


def environment() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "cryptography": cryptography.__version__,
        "machine": platform.machine(),
        "processor": platform.processor() or platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "cpu_count": str(os.cpu_count()),
    }


def run_mode(mode: Mode, runs: int, payload: bytes) -> tuple[list[dict], dict]:
    s = CryptoSuite(mode)
    server_kem, device_sig, verdict_key = s.kem_keygen(), s.sig_keygen(), s.sig_keygen()
    server_pub, device_pub = s.kem_public_bytes(server_kem), s.sig_public_bytes(device_sig)
    kem_ct, ss = s.encapsulate(server_pub)
    nonce, cid = os.urandom(32), "0" * 32
    key = s.derive_key(ss, nonce, cid)
    aad = s.aad(cid)
    digest = sha384(b"manifest")
    sig = s.sign(device_sig, digest)
    iv, ct = s.encrypt(key, payload, aad)
    evidence = Evidence(video=payload, metadata=b"{}", manifest=b"{" + b"x" * 300 + b"}", signature=sig)
    bundle = s.seal(server_pub, cid, nonce, evidence)
    record = build_record(
        challenge_id=cid, mode=mode.value, tx_hash="ab" * 48, sample_hash="cd" * 48,
        scores={k: 0.123456 for k in ("match", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection")},
        model_versions={k: "v" for k in ("face_detector", "matcher", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection")},
        thresholds_version="v", decision="accepted", reason_code=None, timestamp_ms=0, signer="0" * 16,
    )

    ops: dict[str, Callable[[], object]] = {
        "kem_keygen": s.kem_keygen,
        "encapsulate": lambda: s.encapsulate(server_pub),
        "decapsulate": lambda: s.decapsulate(server_kem, kem_ct),
        "sig_keygen": s.sig_keygen,
        "sign_manifest": lambda: s.sign(device_sig, digest),
        "verify_manifest": lambda: s.verify(device_pub, sig, digest),
        "hkdf": lambda: s.derive_key(ss, nonce, cid),
        "aes_gcm_encrypt": lambda: s.encrypt(key, payload, aad),
        "aes_gcm_decrypt": lambda: s.decrypt(key, iv, ct, aad),
        "seal_bundle": lambda: s.seal(server_pub, cid, nonce, evidence),
        "unseal_bundle": lambda: s.unseal(server_kem, bundle, nonce),
        "sign_verdict": lambda: sign_record(mode, verdict_key, record),
    }
    rows = []
    for name, fn in ops.items():
        t = bench(fn, runs)
        rows.append({
            "mode": mode.value, "op": name, "runs": runs,
            "mean_us": round(float(t.mean()), 2), "p50_us": round(float(np.percentile(t, 50)), 2),
            "p95_us": round(float(np.percentile(t, 95)), 2), "min_us": round(float(t.min()), 2),
        })
    sizes = {
        "mode": mode.value,
        "kem_public_key_B": len(server_pub),
        "kem_ciphertext_B": len(kem_ct),
        "sig_public_key_B": len(device_pub),
        "signature_B": len(sig),
        "bundle_overhead_B": len(bundle.to_bytes()) - len(payload),
        "per_transaction_asymmetric_B": len(kem_ct) + len(sig),
    }
    return rows, sizes


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1000)
    ap.add_argument("--payload-kb", type=int, default=2048, help="AES-GCM / seal payload size (a 4 s capture is ~1-3 MB)")
    args = ap.parse_args(argv)
    payload = os.urandom(args.payload_kb * 1024)
    env = environment()
    rows, sizes = [], []
    for mode in (Mode.CLASSICAL, Mode.PQ):
        r, s = run_mode(mode, args.runs, payload)
        rows += [{**x, "payload_kb": args.payload_kb, **env} for x in r]
        sizes.append(s)
    print(f"wrote {write_csv(RESULTS / 'crypto_bench.csv', rows)}")
    print(f"wrote {write_csv(RESULTS / 'crypto_sizes.csv', sizes)}")
    by = {(r["mode"], r["op"]): r for r in rows}
    for op in ("encapsulate", "decapsulate", "sign_manifest", "verify_manifest", "seal_bundle", "unseal_bundle", "sign_verdict"):
        print(f"  {op:16} CLASSICAL p50 {by[('CLASSICAL', op)]['p50_us']:>9} us   PQ p50 {by[('PQ', op)]['p50_us']:>9} us")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
