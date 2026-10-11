"""S6: capture client against a simulated camera and the in-process server."""

from __future__ import annotations

import json
import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from client import capture as capture_mod
from client.capture import capture_session, load_session, measure_resolution_change, save_session
from client.protocol import ApiClient, DeviceIdentity, check_challenge, seal_evidence
from crypto import Mode
from preprocessing.video import encode_jpeg, unpack_jpegs
from server.app import create_app
from tests.conftest import TX, stub_config


class FakeCamera:
    """Behaves like cv2.VideoCapture: ~30 fps, resolution changes apply after a short delay."""

    def __init__(self, width=640, height=480, fps=30.0, switch_delay_s=0.02, honours_resolution=True):
        self.size = (width, height)
        self.pending: tuple[tuple[int, int], float] | None = None
        self.fps = fps
        self.switch_delay_s = switch_delay_s
        self.honours = honours_resolution
        self.n = 0
        self.released = False

    def set(self, prop, value):
        if not self.honours:
            return False
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            self.pending = ((int(value), (self.pending or (self.size, 0))[0][1]), time.perf_counter() + self.switch_delay_s)
        elif prop == cv2.CAP_PROP_FRAME_HEIGHT:
            self.pending = (((self.pending or (self.size, 0))[0][0], int(value)), time.perf_counter() + self.switch_delay_s)
        return True

    def get(self, prop):
        return self.fps if prop == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        time.sleep(1.0 / self.fps)
        if self.pending and time.perf_counter() >= self.pending[1]:
            self.size, self.pending = self.pending[0], None
        self.n += 1
        frame = np.full((self.size[1], self.size[0], 3), self.n % 255, np.uint8)
        return True, frame

    def release(self):
        self.released = True


@pytest.fixture
def config(tmp_path):
    return stub_config(tmp_path, capture={"duration_seconds": 0.4, "resolution_probe": [320, 240]})


def test_resolution_change_latency_measured():
    cam = FakeCamera(switch_delay_s=0.05)
    latency = measure_resolution_change(cam, (320, 240), (640, 480), timeout_s=1.0)
    assert latency is not None and latency >= 40
    assert cam.size == (640, 480)


def test_resolution_change_ignored_by_virtual_camera():
    assert measure_resolution_change(FakeCamera(honours_resolution=False), (320, 240), (640, 480), timeout_s=0.2) is None


def test_capture_session_metadata(config):
    video, meta = capture_session(config, ["blink_twice", "turn_left"], show=False, cap=FakeCamera(), name="Fake Cam")
    jpegs = unpack_jpegs(video)
    assert len(jpegs) == len(meta["frame_timestamps_ns"]) >= 5
    assert meta["frame_timestamps_ns"][0] == 0 and meta["frame_timestamps_ns"] == sorted(meta["frame_timestamps_ns"])
    assert meta["device_name"] == "Fake Cam"
    assert meta["requested"] == {"width": 640, "height": 480, "fps": 30.0}
    assert meta["delivered"]["width"] == 640 and meta["delivered"]["height"] == 480
    assert meta["resolution_change_latency_ms"] is not None


def test_session_save_and_load(tmp_path, config):
    video, meta = capture_session(config, [], show=False, cap=FakeCamera(), name="Fake Cam")
    save_session(tmp_path / "s", video, meta, {"actions": ["turn_left"], "label": "bona_fide"})
    v2, m2 = load_session(tmp_path / "s")
    assert v2 == video and m2 == json.loads(json.dumps(meta))


def test_end_to_end_both_modes(config, tmp_path):
    app = create_app(config)
    with TestClient(app) as http:
        api = ApiClient(http)
        photo = encode_jpeg(np.full((480, 640, 3), 128, np.uint8))
        for mode in Mode:
            identity = api.enroll("carol", mode, photo)
            ch = api.challenge("carol", TX, mode)
            video, meta = capture_session(config, ch["actions"], show=False, cap=FakeCamera(), name="Fake Cam")
            check_challenge(identity, ch, TX)
            resp, rid = api.verify(seal_evidence(identity, ch, video, meta))
            assert resp["decision"] == "accepted", app.state.server.db.get_request(rid)


def test_cli_capture_and_pre_recorded_session(config, tmp_path, monkeypatch, capsys):
    app = create_app(config)
    with TestClient(app) as http:
        api = ApiClient(http)
        identity = api.enroll("dave", Mode.PQ, encode_jpeg(np.full((480, 640, 3), 128, np.uint8)))
        identity.save(config["capture"]["device_dir"])
        monkeypatch.setattr(capture_mod, "ApiClient", lambda url: api)
        monkeypatch.setattr(capture_mod, "open_camera", lambda *a, **k: FakeCamera())
        monkeypatch.setattr(capture_mod, "device_name", lambda i: "Fake Cam")
        monkeypatch.setattr(capture_mod, "load_config", lambda path=None: config)

        argv = ["--user", "dave", "--mode", "PQ", "--tx", json.dumps(TX), "--no-window", "--save", str(tmp_path / "sess")]
        assert capture_mod.main(argv) == 0
        assert "Decision: ACCEPTED" in capsys.readouterr().out

        # The same video sent against a new challenge: stubs accept it, but the server consumed a fresh
        # challenge and signed a separate verdict. (With MediaPipe weights, M4 rejects it as SCRIPT_FAIL.)
        argv = ["--user", "dave", "--mode", "PQ", "--tx", json.dumps(TX), "--from-session", str(tmp_path / "sess")]
        capture_mod.main(argv)
        assert "pre-recorded session" in capsys.readouterr().out


def test_cli_rejects_unknown_identity(config):
    with pytest.raises(FileNotFoundError):
        DeviceIdentity.load(config["capture"]["device_dir"], "nobody", Mode.PQ)
