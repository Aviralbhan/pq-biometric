"""Data preparation, training, calibration, verification and evaluation scripts, on tiny fake datasets."""

from __future__ import annotations

import csv
import json
import os
import random
import shutil
import sqlite3

import cv2
import numpy as np
import pytest
import yaml

import eval.attack_matrix as attack_matrix
import eval.crypto_bench as crypto_bench
import eval.perf as perf
import eval.plots as plots
import scripts.calibrate as calibrate
import scripts.replay_attacks as replay_attacks
import scripts.train_m5 as train_m5
import scripts.verify_verdict as verify_verdict
from client.capture import save_session
from crypto import Mode, canonical_json
from eval.harness import connect, synthetic_session
from scripts import download_frll_morphs, download_openforensics, filter_celeba_spoof
from settings import DEFAULT_CONFIG, load_config
from tests.conftest import TX, stub_config, synthetic_metadata, synthetic_video


def read_csv(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.fixture
def results(tmp_path, monkeypatch):
    out = tmp_path / "results"
    for mod in (attack_matrix, crypto_bench, perf, plots, calibrate, replay_attacks, train_m5):
        monkeypatch.setattr(mod, "RESULTS", out)
    return out


# ---- dataset preparation ----------------------------------------------------------------


def fake_celeba(root, n_ids=6):
    labels = {}
    for split in ("train", "test"):
        for i in range(n_ids):
            ident = f"{split}{i:03d}"
            for kind, spoof_type in (("live", 0), ("spoof", 7), ("spoof", 1)):
                rel = f"Data/{split}/{ident}/{kind}/{ident}_{kind}_{spoof_type}.jpg"
                p = root / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(p), np.zeros((448, 448, 3), np.uint8))
                w = 100 if spoof_type != 1 else 40  # 100/224*448 = 200 px kept; 80 px dropped
                (p.parent / f"{p.stem}_BB.txt").write_text(f"10 10 {w} {w} 0.99")
                lab = [0] * 44
                lab[40], lab[43] = spoof_type, int(kind == "spoof")
                labels[rel] = lab
        (root / "metas" / "intra_test").mkdir(parents=True, exist_ok=True)
        (root / "metas" / "intra_test" / f"{split}_label.json").write_text(json.dumps({k: v for k, v in labels.items() if f"/{split}/" in k}))
    return root


def test_filter_celeba_spoof(tmp_path):
    root = fake_celeba(tmp_path / "cs")
    splits = filter_celeba_spoof.build(root, 112, 0.34, 0, None)
    assert {r["attack_type"] for r in splits["test"]} == {"", "pc"}  # 80 px photo faces dropped
    ids = {k: {r["identity"] for r in v} for k, v in splits.items()}
    assert ids["val"] and not (ids["train"] & ids["val"])
    assert all(r["face_w"] == 200 for rows in splits.values() for r in rows)


def test_filter_celeba_spoof_rejects_wrong_layout(tmp_path):
    root = fake_celeba(tmp_path / "cs")
    path = root / "metas" / "intra_test" / "train_label.json"
    labels = json.loads(path.read_text())
    first = next(k for k in labels if "/live/" in k)
    labels[first][43] = 1
    path.write_text(json.dumps(labels))
    with pytest.raises(SystemExit):
        filter_celeba_spoof.build(root, 112, 0.2, 0, None)


def test_frll_morph_pairs_both_directions_and_three_quarter_probes(tmp_path):
    frll, morphs = tmp_path / "frll", tmp_path / "morphs"
    views = ("neutral_front", "smiling_front", *download_frll_morphs.THREE_QUARTER)
    for view in views:
        (frll / view).mkdir(parents=True)
        for s in range(1, 5):
            (frll / view / f"{s:03d}_03.jpg").write_bytes(b"x")
    (morphs / "facelab_london" / "morph_amsl").mkdir(parents=True)
    (morphs / "facelab_london" / "morph_amsl" / "001_002.jpg").write_bytes(b"x")
    rows = download_frll_morphs.build_pairs(morphs, frll)
    bona = [r for r in rows if r["label"] == "bona_fide"]
    assert len(bona) == 4 * 10
    assert {(r["reference_view"], r["probe_view"]) for r in bona if r["subject_a"] == "001"} == (
        {("neutral_front", v) for v in views if v != "neutral_front"} | {("smiling_front", v) for v in views if v != "smiling_front"})
    morph = [r for r in rows if r["label"] == "morph"]
    assert len(morph) == 2 * 5 and {r["morph_method"] for r in morph} == {"amsl"}
    assert "neutral_front" not in {r["probe_view"] for r in morph}


def test_m2_folds_group_subjects():
    import scripts.train_m2 as train_m2

    subjects = [f"{i:03d}" for i in range(1, 103)]
    fold = train_m2.fold_of_subjects(subjects, 5, 0)
    assert set(fold) == set(subjects) and sorted(set(fold.values())) == [0, 1, 2, 3, 4]
    sizes = sorted(list(fold.values()).count(k) for k in range(5))
    assert sizes[0] >= 20 and sizes[-1] <= 21


def test_openforensics_crops(tmp_path):
    images = tmp_path / "images"
    images.mkdir()
    cv2.imwrite(str(images / "a.jpg"), np.full((600, 800, 3), 120, np.uint8))
    coco = {
        "categories": [{"id": 0, "name": "Real"}, {"id": 1, "name": "Fake"}],
        "images": [{"id": 1, "file_name": "Val/a.jpg"}],
        "annotations": [
            {"id": 10, "image_id": 1, "bbox": [100, 100, 200, 200], "category_id": 0},
            {"id": 11, "image_id": 1, "bbox": [400, 100, 150, 150], "category_id": 1},
            {"id": 12, "image_id": 1, "bbox": [600, 400, 50, 50], "category_id": 1},
        ],
    }
    ann = tmp_path / "ann.json"
    ann.write_text(json.dumps(coco))
    rows = download_openforensics.crop_split("Val", images, ann, tmp_path / "crops", 112, None)
    assert [(r["label"], r["attack_type"]) for r in rows] == [("bona_fide", ""), ("attack", "openforensics_fake")]
    assert cv2.imread(rows[0]["path"]).shape == (224, 224, 3)


# ---- M5 training ---------------------------------------------------------------------------


def test_train_m5_leave_one_tool_out(tmp_path, results, monkeypatch):
    rng = random.Random(0)
    sessions = tmp_path / "m5"
    for label, jitter, name in (("bona_fide", 4.0, "Integrated Webcam"), ("obs", 0.05, "OBS Virtual Camera"), ("v4l2loopback", 0.1, "Dummy video device")):
        for subject in ("aviral", "krish"):
            for i in range(12):
                meta = synthetic_metadata(n=20, device=name)
                meta["frame_timestamps_ns"] = [int(k * 33.3e6 + rng.gauss(0, jitter * 1e6)) for k in range(20)]
                save_session(sessions / label / f"{subject}_{i:03d}", synthetic_video(20, seed=i), meta, {"label": label})
    model_path = tmp_path / "m5.joblib"
    cfg = stub_config(tmp_path, detectors={"m5_injection": {"model": str(model_path)}})
    monkeypatch.setattr(train_m5, "load_config", lambda path=None: cfg)
    assert train_m5.main(["--sessions", str(sessions)]) == 0
    rows = read_csv(results / "m5_injection.csv")
    assert {r["fold"] for r in rows} == {"held_out=obs", "held_out=v4l2loopback"}
    assert all(float(r["auc"]) > 0.9 for r in rows)
    assert model_path.exists()


# ---- calibration ----------------------------------------------------------------------------


def test_calibrate_writes_thresholds(tmp_path, results):
    results.mkdir()
    rng = np.random.default_rng(0)
    with open(results / "scores_m1_pad_zero_shot.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "label", "attack_type", "score"])
        for s in rng.uniform(0, 0.3, 1000):
            w.writerow(["val", "bona_fide", "", s])
        w.writerow(["val", "attack", "pc", 0.9])
    with open(results / "matcher_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["split", "kind", "similarity"])
        for s in rng.normal(0.0, 0.1, 5000):
            w.writerow(["val", "impostor", s])
    cfg_path = tmp_path / "config.yaml"
    shutil.copy(DEFAULT_CONFIG, cfg_path)
    assert calibrate.main(["--config", str(cfg_path)]) == 0
    cfg = yaml.safe_load(cfg_path.read_text())
    assert 0.29 < cfg["thresholds"]["m1_pad"] <= 0.3
    assert cfg["thresholds"]["m2_morph"] == yaml.safe_load(DEFAULT_CONFIG.read_text())["thresholds"]["m2_morph"]  # no scores: unchanged
    assert 0.2 < cfg["thresholds"]["match"] < 0.45
    assert cfg["thresholds"]["version"].startswith("cal-")
    assert cfg["detectors"]["m5_injection"]["virtual_camera_names"]  # rest of the file intact
    assert calibrate.main(["--config", str(cfg_path)]) == 0  # idempotent: the block is replaced, not appended
    assert cfg_path.read_text().count("thresholds:") == 1


# ---- verdict verification --------------------------------------------------------------------


def test_verify_verdict_cli(tmp_path):
    cfg = stub_config(tmp_path)
    video, meta, ref = synthetic_session(6)
    with connect(cfg) as conn:
        for mode in Mode:
            identity = conn.api.enroll(f"v-{mode.value}", mode, ref)
            conn.api.authenticate(identity, TX, video, meta)
        conn.server.db.close()
    cfg_file = tmp_path / "cfg.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg))
    assert verify_verdict.main(["--all", "--config", str(cfg_file)]) == 0
    with sqlite3.connect(cfg["server"]["db_path"]) as conn:
        cid, record = conn.execute("SELECT challenge_id, record FROM verdicts WHERE mode = 'PQ'").fetchone()
        edited = json.loads(record)
        edited["scores"]["m3_deepfake"] = 0.0 if edited["scores"]["m3_deepfake"] else 0.9
        conn.execute("UPDATE verdicts SET record = ? WHERE challenge_id = ?", (canonical_json(edited), cid))
    assert verify_verdict.main(["--challenge-id", cid, "--config", str(cfg_file)]) == 1
    assert verify_verdict.main(["--all", "--config", str(cfg_file)]) == 1


# ---- evaluation harness -------------------------------------------------------------------------


def test_replay_suite_small(tmp_path, results):
    cfg = stub_config(tmp_path, challenge={"ttl_seconds": 1})
    video, meta, ref = synthetic_session(6)
    with connect(cfg) as conn:
        identity = conn.api.enroll("r", Mode.PQ, ref)
        rows = replay_attacks.run(conn, identity, video, meta, 3, 0, random.Random(0), log=lambda *a: None)
    by = {r["case"]: r for r in rows}
    for case, code in replay_attacks.EXPECTED.items():
        assert by[case]["correct_code"] == 3, (case, by[case])
    assert by["genuine"]["protocol_false_rejections"] == 0


def test_crypto_bench_small():
    rows, sizes = crypto_bench.run_mode(Mode.PQ, 5, b"x" * 1024)
    assert {r["op"] for r in rows} >= {"encapsulate", "sign_manifest", "seal_bundle", "sign_verdict"}
    assert sizes["kem_ciphertext_B"] == 1088 and sizes["signature_B"] == 3309


def test_perf_small(tmp_path):
    cfg = stub_config(tmp_path)
    video, meta, ref = synthetic_session(6)
    with connect(cfg) as conn:
        stages, summary = perf.run_mode(conn, Mode.CLASSICAL, 4, video, meta, ref, os.getpid())
    names = {s["stage"] for s in stages}
    assert {"end_to_end", "client_seal", "server_check3_unseal", "server_total"} <= names
    assert summary["outcomes"] == "accepted:None=4" and summary["bytes_per_request_mean"] > len(video)


def test_attack_matrix_and_plots(tmp_path, results, monkeypatch):
    imgs = tmp_path / "print"
    imgs.mkdir()
    for i in range(2):
        cv2.imwrite(str(imgs / f"{i}.jpg"), np.full((300, 300, 3), 60 * (i + 1), np.uint8))
    obs = tmp_path / "obs"
    save_session(obs / "s0", synthetic_video(6), synthetic_metadata(6, device="OBS Virtual Camera"))
    cfg = stub_config(tmp_path)
    monkeypatch.setattr(attack_matrix, "load_config", lambda path=None: cfg)
    assert attack_matrix.main(["--attack", f"print={imgs}", "--attack", f"obs={obs}", "--modes", "PQ", "CLASSICAL"]) == 0
    rows = {(r["mode"], r["attack"]): r for r in read_csv(results / "attack_matrix.csv")}
    assert float(rows[("PQ", "obs")]["rejection_rate"]) == 1.0 and "INJECTION_FAIL" in rows[("PQ", "obs")]["reason_codes"]
    assert float(rows[("PQ", "print")]["rejection_rate"]) == 0.0  # stubs cannot see a print attack
    assert plots.main() == 0
    assert (results / "figures" / "attack_matrix.png").exists()


def test_default_config_loads():
    cfg = load_config()
    assert set(cfg["thresholds"]) >= {"version", "match", "m1_pad", "m2_morph", "m3_deepfake", "m4_replay", "m5_injection"}
    assert len(cfg["detectors"]["m1_pad"]["models"]) == len(cfg["preprocessing"]["pad_scales"])
