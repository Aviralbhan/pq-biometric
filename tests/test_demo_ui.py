"""Demo UI (review use only): only mounted in demo mode, local only, friendly errors, no backend changes."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from server.app import create_app
from tests.conftest import stub_config


@pytest.fixture
def demo_client(tmp_path, monkeypatch):
    monkeypatch.setenv("PQBIO_DEMO_UI", "1")
    app = create_app(stub_config(tmp_path, server={"host": "127.0.0.1"}))
    with TestClient(app) as http:
        yield http


def test_ui_absent_without_demo_mode(tmp_path, monkeypatch):
    monkeypatch.delenv("PQBIO_DEMO_UI", raising=False)
    with TestClient(create_app(stub_config(tmp_path))) as http:
        assert http.get("/ui/").status_code == 404
        assert http.get("/demo/api/status").status_code == 404


def test_demo_mode_refuses_non_local_host(tmp_path, monkeypatch):
    monkeypatch.setenv("PQBIO_DEMO_UI", "1")
    with pytest.raises(RuntimeError):
        create_app(stub_config(tmp_path, server={"host": "0.0.0.0"}))


def test_page_is_served_without_external_links(demo_client):
    assert demo_client.get("/ui", follow_redirects=False).status_code in (302, 307)
    page = demo_client.get("/ui/")
    assert page.status_code == 200 and "Post-Quantum" in page.text
    for asset in ("/ui/", "/ui/app.js", "/ui/style.css"):
        body = demo_client.get(asset).text
        assert "http://" not in body.replace("http://127.0.0.1", "") and "https://" not in body  # works offline


def test_status_reports_stubs_and_preliminary_thresholds(demo_client):
    s = demo_client.get("/demo/api/status").json()
    assert set(s["stubs"]) >= {"matcher", "m1_pad", "m3_deepfake"}  # stub config: no weights
    assert {"PQ", "CLASSICAL"} <= set(s["allowed_modes"])
    assert "version" in s["thresholds"] and s["busy"] is False


def test_results_lists_every_table(demo_client):
    r = demo_client.get("/demo/api/results").json()
    assert set(r) == {"matcher", "m2", "m3_zero_shot", "m3_compare", "replay", "genuine"}
    for table in r.values():
        assert table is None or ("rows" in table and "columns" in table)  # None = pending


def test_steps_fail_friendly(demo_client):
    assert demo_client.post("/demo/api/run/nope", json={"mode": "PQ"}).json() == {"ok": False, "message": "Unknown step 'nope'."}
    assert demo_client.post("/demo/api/run/verify", json={"mode": "HYBRID"}).status_code == 400
    r = demo_client.post("/demo/api/run/deepfake", json={"mode": "PQ"}).json()
    assert r["ok"] is False and "stub" in r["message"] and "Traceback" not in r["message"]


def test_crypto_step_pending_or_table(demo_client):
    r = demo_client.post("/demo/api/run/crypto", json={"mode": "PQ"}).json()
    assert r["ok"] is True and (r["pending"] is True or len(r["ops"]) == 7)
