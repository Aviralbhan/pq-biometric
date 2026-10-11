"""Verification server (Layer C).

POST /enroll     register a device public key for (user, mode) and the face template
POST /challenge  issue a single-use challenge bound to a transaction
POST /verify     sealed evidence bundle (application/octet-stream) -> accepted / rejected
GET  /health     modes and module versions

The client only ever sees accepted/rejected and the challenge_id. Reason codes go
to the server log and the SQLite request log; the signed verdict holds the rest.
"""

from __future__ import annotations

import base64
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from crypto import Mode
from detectors import build_all
from preprocessing.faces import FacePreprocessor
from preprocessing.video import decode_jpeg
from server.challenge import ChallengeStore, make_redis
from server.checks import run_checks
from server.decision import SCORE_KEYS, decide
from server.pipeline import Pipeline
from server.store import Database, ServerKeys, now_ms
from server.timing import Timer
from server.verdict import build_record, sign_record, signer_id
from settings import load_config, resolve

log = logging.getLogger("pqbio.server")

USER_ID_PATTERN = r"^[A-Za-z0-9_.\-]{1,64}$"


class EnrollRequest(BaseModel):
    user_id: str = Field(pattern=USER_ID_PATTERN)
    mode: Mode
    device_pub: str  # base64
    reference_jpeg: str  # base64


class ChallengeRequest(BaseModel):
    user_id: str = Field(pattern=USER_ID_PATTERN)
    tx_context: dict[str, Any]
    mode: Mode | None = None


def b64e(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64d(text: str) -> bytes:
    try:
        return base64.b64decode(text, validate=True)
    except ValueError as exc:
        raise HTTPException(400, "invalid base64") from exc


class Server:
    """All server state, built once from the config."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        s = config["server"]
        self.keys = ServerKeys(resolve(s["keys_dir"]))
        self.db = Database(resolve(s["db_path"]), self.keys.storage_key)
        self.store = ChallengeStore(make_redis(config["redis"]["url"]), config, self.keys.action_secret)
        self.default_mode = Mode(config["crypto"]["mode"])
        self.allowed_modes = {Mode(m) for m in config["crypto"]["allowed_modes"]} | {self.default_mode}
        self.thresholds = config["thresholds"]
        self.preprocessor = FacePreprocessor.from_config(config)
        self.detectors = build_all(config)
        self.executor = ThreadPoolExecutor(max_workers=len(self.detectors) + 1, thread_name_prefix="det")
        self.pipeline = Pipeline(self.preprocessor, self.detectors, config["preprocessing"]["sample_frames"], self.executor)
        self.max_body = int(s["max_body_bytes"])

    def close(self) -> None:
        self.executor.shutdown(wait=False)
        self.db.close()

    # ------------------------------------------------------------------ enrol

    def enroll(self, req: EnrollRequest) -> dict[str, Any]:
        if req.mode not in self.allowed_modes:
            raise HTTPException(400, "mode not allowed")
        device_pub = b64d(req.device_pub)
        if len(device_pub) != {Mode.PQ: 1952, Mode.CLASSICAL: 32}[req.mode]:
            raise HTTPException(400, "device public key has the wrong length for this mode")
        try:
            image = decode_jpeg(b64d(req.reference_jpeg))
        except ValueError as exc:
            raise HTTPException(400, "reference image is not a JPEG") from exc
        crops = self.preprocessor.process(image)
        if crops is None:
            raise HTTPException(422, "no face found in the reference image")
        template = self.detectors["matcher"].embed(crops.arcface112)
        self.db.put_template(req.user_id, template, self.detectors["matcher"].version)
        self.db.put_device_key(req.user_id, req.mode, device_pub)
        log.info("ENROLL user=%s mode=%s", req.user_id, req.mode.value)
        return {
            "user_id": req.user_id,
            "mode": req.mode.value,
            "server_kem_pub": b64e(self.keys.kem_public(req.mode)),
            "verdict_pub": b64e(self.keys.verdict_public(req.mode)),
        }

    # -------------------------------------------------------------- challenge

    def challenge(self, req: ChallengeRequest) -> dict[str, Any]:
        mode = req.mode or self.default_mode
        if mode not in self.allowed_modes:
            raise HTTPException(400, "mode not allowed")
        if self.db.get_device_key(req.user_id, mode) is None:
            raise HTTPException(404, "user not enrolled for this mode")
        ch = self.store.issue(req.user_id, mode, req.tx_context)
        return {
            "challenge_id": ch.challenge_id,
            "mode": ch.mode,
            "nonce": b64e(ch.nonce),
            "tx_hash": ch.tx_hash,
            "actions": ch.actions,
            "server_kem_pub": b64e(self.keys.kem_public(mode)),
            "expires_ms": ch.expires_ms,
        }

    # ----------------------------------------------------------------- verify

    async def verify(self, body: bytes) -> tuple[dict[str, Any], int]:
        timer = Timer()
        t0 = time.perf_counter_ns()
        res = run_checks(body, self.store, self.keys, self.db, timer)
        scores: dict[str, float | None] = {k: None for k in SCORE_KEYS}
        reason = res.reason
        accepted = False

        if res.ok:
            stored = self.db.get_template(res.challenge.user_id)
            template = stored[0] if stored else None
            scores.update(await self.pipeline.analyse(res.evidence, res.challenge.actions, template, timer))
            with timer.stage("decision"):
                accepted, reason = decide(scores, self.thresholds)

        decision = "accepted" if accepted else "rejected"
        ch = res.challenge
        if ch is not None:  # this request consumed the challenge: sign exactly one verdict for it
            mode = Mode(ch.mode)
            with timer.stage("verdict_sign"):
                record = build_record(
                    challenge_id=ch.challenge_id,
                    mode=ch.mode,
                    tx_hash=ch.tx_hash,
                    sample_hash=res.manifest.sample_hash if res.manifest else None,
                    scores=scores,
                    model_versions=self.pipeline.versions(),
                    thresholds_version=str(self.thresholds.get("version")),
                    decision=decision,
                    reason_code=reason.value if reason else None,
                    timestamp_ms=now_ms(),
                    signer=signer_id(self.keys.verdict_public(mode)),
                )
                record_bytes, signature = sign_record(mode, self.keys.verdict_private(mode), record)
            self.db.put_verdict(ch.challenge_id, mode, record_bytes, signature)

        timer.add("total", (time.perf_counter_ns() - t0) / 1000.0)
        bundle = res.bundle
        challenge_id = bundle.challenge_id if bundle else None
        request_id = self.db.add_request(
            challenge_id, bundle.mode.value if bundle else None, decision, reason.value if reason else None, len(body), timer.us
        )
        log.info(
            "VERIFY request=%d challenge=%s mode=%s decision=%s reason=%s",
            request_id, challenge_id, bundle.mode.value if bundle else None, decision, reason.value if reason else "-",
        )
        return {"decision": decision, "challenge_id": challenge_id}, request_id


def _configure_logging(path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and h.baseFilename == str(path) for h in log.handlers):
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)


def create_app(config: dict[str, Any] | None = None) -> FastAPI:
    config = config or load_config()
    _configure_logging(resolve(config["server"]["log_path"]))
    server = Server(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        server.close()

    app = FastAPI(title="PQ biometric evidence authentication", lifespan=lifespan)
    app.state.server = server

    @app.get("/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "default_mode": server.default_mode.value,
            "allowed_modes": sorted(m.value for m in server.allowed_modes),
            "model_versions": server.pipeline.versions(),
            "thresholds_version": server.thresholds.get("version"),
        }

    @app.post("/enroll")
    def enroll(req: EnrollRequest) -> dict[str, Any]:
        return server.enroll(req)

    @app.post("/challenge")
    def challenge(req: ChallengeRequest) -> dict[str, Any]:
        return server.challenge(req)

    @app.post("/verify")
    async def verify(request: Request) -> JSONResponse:
        body = await request.body()
        if len(body) > server.max_body:
            raise HTTPException(413, "request too large")
        payload, request_id = await server.verify(body)
        return JSONResponse(payload, headers={"X-Request-Id": str(request_id)})

    return app


def __getattr__(name: str):
    # `uvicorn server.app:app` builds the app from config.yaml on first access.
    if name == "app":
        globals()["app"] = create_app()
        return globals()["app"]
    raise AttributeError(name)

