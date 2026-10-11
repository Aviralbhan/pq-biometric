"""End-to-end performance per mode: per-stage latency, bytes, CPU and memory.

    python -m eval.perf --runs 500 [--session DIR] [--url http://127.0.0.1:8000 --server-pid PID]

Each run is one genuine transaction: challenge -> seal on the client -> verify.
Server stage times come from the request log (time.perf_counter_ns inside the
server); client times are measured here. CPU % and RSS are sampled every 50 ms with
psutil from the server process (--server-pid) or, in-process, from this process.

Writes eval/results/perf_stages.csv and eval/results/perf_summary.csv.
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from collections import defaultdict

import numpy as np
import psutil

from client.protocol import check_challenge, seal_evidence
from crypto import Mode
from eval.harness import RESULTS, connect, percentile, session_or_synthetic, write_csv
from settings import load_config

TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}
SERVER_CRYPTO_STAGES = ("check3_unseal", "check6_signature", "verdict_sign")


class ResourceSampler:
    def __init__(self, pid: int, interval_s: float = 0.05):
        self.proc = psutil.Process(pid)
        self.interval = interval_s
        self.cpu: list[float] = []
        self.rss: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        self.proc.cpu_percent(None)
        while not self._stop.wait(self.interval):
            self.cpu.append(self.proc.cpu_percent(None))
            self.rss.append(self.proc.memory_info().rss / 2**20)

    def __enter__(self) -> ResourceSampler:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()


def run_mode(conn, mode: Mode, runs: int, video: bytes, meta: dict, ref: bytes, pid: int) -> tuple[list[dict], dict]:
    api = conn.api
    identity = api.enroll(f"perf-{mode.value.lower()}", mode, ref)
    stages: dict[str, list[float]] = defaultdict(list)
    bytes_in, decisions = [], defaultdict(int)
    for _ in range(3):  # warm-up: model sessions, connection pool
        api.authenticate(identity, TX, video, meta)

    with ResourceSampler(pid) as res:
        t_start = time.perf_counter()
        for _ in range(runs):
            t0 = time.perf_counter_ns()
            ch = api.challenge(identity.user_id, TX, mode)
            t1 = time.perf_counter_ns()
            check_challenge(identity, ch, TX)
            bundle = seal_evidence(identity, ch, video, meta)
            t2 = time.perf_counter_ns()
            resp, rid = api.verify(bundle)
            t3 = time.perf_counter_ns()
            req = conn.request(rid)
            decisions[f"{resp['decision']}:{req['reason_code']}"] += 1
            bytes_in.append(req["bytes_in"])
            stages["client_challenge_rtt"].append((t1 - t0) / 1e3)
            stages["client_seal"].append((t2 - t1) / 1e3)
            stages["client_verify_rtt"].append((t3 - t2) / 1e3)
            stages["end_to_end"].append((t3 - t0) / 1e3)
            for k, us in req["timing"].items():
                stages[f"server_{k}"].append(us)
            stages["server_crypto"].append(sum(req["timing"].get(k, 0.0) for k in SERVER_CRYPTO_STAGES))
        elapsed = time.perf_counter() - t_start

    stage_rows = [
        {
            "mode": mode.value, "stage": k, "n": len(v),
            "mean_ms": round(float(np.mean(v)) / 1e3, 4),
            "p50_ms": round(percentile(v, 50) / 1e3, 4),
            "p95_ms": round(percentile(v, 95) / 1e3, 4),
        }
        for k, v in sorted(stages.items())
    ]
    crypto = np.array(stages["server_crypto"]) + np.array(stages["client_seal"])
    share = crypto / np.array(stages["end_to_end"])
    summary = {
        "mode": mode.value,
        "runs": runs,
        "end_to_end_p50_ms": round(percentile(stages["end_to_end"], 50) / 1e3, 3),
        "end_to_end_p95_ms": round(percentile(stages["end_to_end"], 95) / 1e3, 3),
        "server_total_p50_ms": round(percentile(stages["server_total"], 50) / 1e3, 3),
        "server_total_p95_ms": round(percentile(stages["server_total"], 95) / 1e3, 3),
        "crypto_p50_ms": round(percentile(crypto, 50) / 1e3, 3),
        "crypto_share_p50": round(percentile(share, 50), 5),
        "bytes_per_request_mean": round(float(np.mean(bytes_in)), 1),
        "sequential_req_per_s": round(runs / elapsed, 3),
        "cpu_percent_mean": round(float(np.mean(res.cpu)), 1) if res.cpu else None,
        "cpu_percent_p95": round(percentile(res.cpu, 95), 1) if res.cpu else None,
        "rss_peak_mb": round(max(res.rss), 1) if res.rss else None,
        "outcomes": ";".join(f"{k}={v}" for k, v in sorted(decisions.items())),
    }
    return stage_rows, summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=500)
    ap.add_argument("--modes", nargs="+", default=["CLASSICAL", "PQ"], choices=[m.value for m in Mode])
    ap.add_argument("--session")
    ap.add_argument("--reference")
    ap.add_argument("--url")
    ap.add_argument("--server-pid", type=int)
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    if args.url and not args.server_pid:
        ap.error("--url needs --server-pid so CPU and memory are measured on the server")

    config = load_config(args.config)
    video, meta, ref, source = session_or_synthetic(args.session, args.reference)
    stage_rows, summaries = [], []
    with connect(config, args.url) as conn:
        for m in args.modes:
            s_rows, summary = run_mode(conn, Mode(m), args.runs, video, meta, ref, args.server_pid or os.getpid())
            stage_rows += s_rows
            summaries.append({**summary, "video": source, "video_bytes": len(video), "in_process": not args.url})
    print(f"wrote {write_csv(RESULTS / 'perf_stages.csv', stage_rows)}")
    print(f"wrote {write_csv(RESULTS / 'perf_summary.csv', summaries)}")
    for s in summaries:
        print(f"  {s['mode']:9} e2e p50 {s['end_to_end_p50_ms']} ms  p95 {s['end_to_end_p95_ms']} ms  crypto share {s['crypto_share_p50']:.2%}  {s['bytes_per_request_mean']:.0f} B/request")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
