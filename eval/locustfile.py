"""Throughput under concurrent load (locust).

    uvicorn server.app:app --port 8000 --workers 1
    PQBIO_MODE=PQ locust -f eval/locustfile.py --host http://127.0.0.1:8000 \
        --headless -u 8 -r 2 -t 2m --csv eval/results/locust_pq
    PQBIO_MODE=CLASSICAL locust ... --csv eval/results/locust_classical

Each simulated user enrols once, then loops: challenge -> seal -> verify.
PQBIO_SESSION=<dir> sends a recorded session instead of the synthetic video.
Locust's own CSVs (<prefix>_stats.csv) give requests/s and latency percentiles.
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from locust import HttpUser, between, task  # noqa: E402

from client.protocol import ApiClient, check_challenge, seal_evidence  # noqa: E402
from crypto import Mode  # noqa: E402
from eval.harness import session_or_synthetic  # noqa: E402

MODE = Mode(os.environ.get("PQBIO_MODE", "PQ"))
VIDEO, META, REF, _ = session_or_synthetic(os.environ.get("PQBIO_SESSION"))
TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}


class Authenticator(HttpUser):
    wait_time = between(0.0, 0.1)

    def on_start(self) -> None:
        # Enrol with a plain httpx client so enrolment is not counted in the stats.
        self.identity = ApiClient(self.host).enroll(f"load-{uuid.uuid4().hex[:12]}", MODE, REF)

    @task
    def authenticate(self) -> None:
        ch = self.client.post("/challenge", json={"user_id": self.identity.user_id, "tx_context": TX, "mode": MODE.value}, name="/challenge").json()
        check_challenge(self.identity, ch, TX)
        body = seal_evidence(self.identity, ch, VIDEO, META).to_bytes()
        with self.client.post(
            "/verify", data=body, headers={"Content-Type": "application/octet-stream"}, name=f"/verify [{MODE.value}]", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
