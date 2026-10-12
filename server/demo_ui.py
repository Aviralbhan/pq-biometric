"""Demo UI (review use only): the /ui page and the /demo/api endpoints behind it.

Mounted by server.app only when PQBIO_DEMO_UI=1 (set by `python scripts/demo.py server`) and the
server is bound to localhost. Every step calls the same functions as the CLI in scripts/demo.py;
nothing here changes the checks, thresholds, models or crypto. One step runs at a time.
"""

from __future__ import annotations

import asyncio
import csv
import logging
import shutil
from pathlib import Path
from typing import Any

import cv2
from fastapi import APIRouter, FastAPI
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from settings import ROOT

log = logging.getLogger("pqbio.demo_ui")
STATIC = Path(__file__).resolve().parent / "static" / "demo"
RESULTS = ROOT / "eval" / "results"
RESULT_FILES = {
    "matcher": "matcher.csv",
    "m2": "m2_morph.csv",
    "m3_zero_shot": "m3_deepfake_zero_shot.csv",
    "m3_compare": "m3_zero_shot_vs_ln_tuned.csv",
    "replay": "replay.csv",
    "genuine": "genuine_webcam_captures.csv",
}
FRIENDLY = {
    FileNotFoundError: "Something this step needs is missing.",
    LookupError: "This step needs an earlier step first.",
}


class RunRequest(BaseModel):
    mode: str = "PQ"


def _demo():
    import scripts.demo as demo  # imported lazily: it is a script module with its own path setup

    return demo


def _request_for(challenge_id: str | None) -> dict | None:
    if not challenge_id:
        return None
    demo = _demo()
    with demo.db() as conn:
        row = conn.execute("SELECT MAX(request_id) FROM requests WHERE challenge_id = ?", (challenge_id,)).fetchone()
    return demo.request_data(int(row[0])) if row and row[0] else None


def _read_csv(name: str) -> dict | None:
    path = RESULTS / name
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    return {"file": name, "columns": reader.fieldnames or [], "rows": rows}


def mount(app: FastAPI, server: Any) -> None:
    router = APIRouter(prefix="/demo/api")
    lock = asyncio.Lock()

    def embed(path: str):
        """The server's own face preprocessing and matcher (same models and settings as scripts/train_m2.Embedder)."""
        img = cv2.imread(path)
        crops = server.preprocessor.process(img) if img is not None else None
        return server.detectors["matcher"].embed(crops.arcface112) if crops else None

    def verify_result(mode: str, camera: int | None = None) -> dict:
        demo = _demo()
        code, challenge_id = demo.verify_capture(mode, camera)
        verdict = demo.verdict_data(challenge_id) if challenge_id else None
        if verdict is None:
            return {"ok": False, "message": "The capture did not reach the server (was the webcam window closed?)."}
        return {"ok": True, "verdict": verdict, "request": _request_for(challenge_id)}

    def run_step(step: str, mode: str) -> dict:
        demo = _demo()
        if step == "enrol":
            demo.demo_config()
            code = demo.run("-m", "client.enroll", "--user", demo.USER, "--config", str(demo.CONFIG))
            return {"ok": code == 0, "message": "Enrolled for PQ and CLASSICAL." if code == 0 else
                    "Enrolment did not finish (no face found, or the window was closed)."}
        if step in ("verify", "photo"):
            return verify_result(mode)
        if step == "replay":
            rows = demo.replay_data()
            for row in rows:
                row["request"] = demo.request_data(row["request_id"])
            return {"ok": True, "attacks": rows}
        if step == "morph":
            return {"ok": True, **demo.morph_data(embed)}
        if step == "deepfake":
            m3 = server.detectors["m3_deepfake"]
            if not hasattr(m3, "p_fake"):
                return {"ok": False, "message": "M3 is running as a stub (no GenD weights)."}
            return {"ok": True, **demo.deepfake_data(m3)}
        if step == "injection":
            obs = demo.obs_camera()
            if obs is None:
                return {"ok": False, "skipped": True, "message": "OBS not installed: step skipped."}
            return verify_result("PQ", obs[0])
        if step == "verdict":
            return {"ok": True, **demo.verdict_tamper_data()}
        if step == "crypto":
            data = demo.crypto_data()
            return {"ok": True, "pending": data is None, **(data or {})}
        return {"ok": False, "message": f"Unknown step {step!r}."}

    @router.get("/status")
    def status() -> dict:
        demo = _demo()
        versions = server.pipeline.versions()
        try:
            obs = demo.obs_camera()
        except Exception:  # camera enumeration is best effort
            obs = None
        return {
            "model_versions": versions,
            "stubs": sorted(k for k, v in versions.items() if v == "stub"),
            "default_mode": server.default_mode.value,
            "allowed_modes": sorted(m.value for m in server.allowed_modes),
            "thresholds": server.thresholds,
            "obs_camera": obs[1] if obs else None,
            "has_genuine_bundle": (demo.LAST / "bundle.bin").exists(),
            "crypto_available": demo.crypto_data() is not None,
            "busy": lock.locked(),
        }

    @router.post("/run/{step}")
    async def run(step: str, req: RunRequest) -> JSONResponse:
        if lock.locked():
            return JSONResponse({"ok": False, "message": "Another step is still running."}, status_code=409)
        if req.mode not in ("PQ", "CLASSICAL"):
            return JSONResponse({"ok": False, "message": "Mode must be PQ or CLASSICAL."}, status_code=400)
        async with lock:
            try:
                # Webcam steps start the client, which calls back into this server: run it off the event loop.
                return JSONResponse(await asyncio.to_thread(run_step, step, req.mode))
            except Exception as exc:
                log.exception("demo step %s failed", step)
                message = next((m for t, m in FRIENDLY.items() if isinstance(exc, t)), "This step failed.")
                return JSONResponse({"ok": False, "message": f"{message} {exc}".strip()})

    @router.get("/results")
    def results() -> dict:
        return {key: _read_csv(name) for key, name in RESULT_FILES.items()}

    @router.post("/reset")
    async def reset() -> JSONResponse:
        """Clear the demo's enrolment, history and saved sessions. Server keys and models stay loaded."""
        if lock.locked():
            return JSONResponse({"ok": False, "message": "Another step is still running."}, status_code=409)
        async with lock:
            demo = _demo()
            for name in ("device", "last", "attempt"):
                shutil.rmtree(demo.DEMO / name, ignore_errors=True)
            (demo.DEMO / "tampered_copy.sqlite").unlink(missing_ok=True)
            for table in ("enrolments", "templates", "verdicts", "requests"):
                server.db._exec(f"DELETE FROM {table}")
            server.store.r.flushall()
            return JSONResponse({"ok": True, "message": "Demo reset: enrol again to start."})

    app.include_router(router)

    @app.get("/ui", include_in_schema=False)
    def ui_redirect() -> RedirectResponse:
        return RedirectResponse("/ui/")

    app.mount("/ui/", StaticFiles(directory=STATIC, html=True), name="demo-ui")
