"""Detector wrappers, tested with tiny generated ONNX graphs and synthetic signals (no real weights)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from detectors.base import Sample, Stub, softmax
from detectors.m1_pad.minifasnet import MiniFASNetPAD
from detectors.m2_morph.dmad import DMADMorph, dmad_feature, save as save_m2, train_svm
from detectors.m3_deepfake.gend import CLIP_MEAN, CLIP_STD, GenDDeepfake, preprocess as gend_preprocess
from detectors.m4_replay.actions import Event, FrameSignal, actions_in_order, detect_events, frame_times_ms, subsample, yaw_from_matrix
from detectors.m5_injection.classifier import InjectionDetector, save as save_m5, train as train_m5
from detectors.m5_injection.features import FEATURE_NAMES, extract, is_virtual_name, to_vector
from detectors.matcher.arcface import ArcFaceMatcher, MatcherStub, cosine, preprocess as arc_preprocess
from preprocessing.faces import CenterFaceStub, FacePreprocessor
from preprocessing.video import encode_jpeg
from server.checks import Reason
from server.decision import decide
from tests.conftest import synthetic_metadata


def linear_onnx(path: Path, in_shape: list[int], weight: np.ndarray, bias: np.ndarray) -> Path:
    """ONNX graph: flatten(x) @ weight + bias. Enough to check pre/post-processing of a wrapper."""
    x = helper.make_tensor_value_info("input", TensorProto.FLOAT, ["N", *in_shape[1:]])
    y = helper.make_tensor_value_info("output", TensorProto.FLOAT, ["N", weight.shape[1]])
    nodes = [
        helper.make_node("Flatten", ["input"], ["flat"], axis=1),
        helper.make_node("Gemm", ["flat", "W", "B"], ["output"]),
    ]
    inits = [
        helper.make_tensor("W", TensorProto.FLOAT, weight.shape, weight.astype(np.float32).ravel()),
        helper.make_tensor("B", TensorProto.FLOAT, bias.shape, bias.astype(np.float32).ravel()),
    ]
    model = helper.make_model(helper.make_graph(nodes, "g", [x], [y], inits), opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.save(model, path)
    return path


def face_sample(n_frames: int = 3, **kw) -> Sample:
    pre = FacePreprocessor(CenterFaceStub(), min_face_px=112)
    rng = np.random.default_rng(1)
    frames = [rng.integers(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(n_frames)]
    crops = [pre.process(f) for f in frames]
    return Sample(jpegs=[], metadata={}, actions=[], frames=frames, crops=crops, best=crops[0], **kw)


# ---- matcher ------------------------------------------------------------------


def test_arcface_preprocess_range_and_layout():
    crop = np.zeros((112, 112, 3), np.uint8)
    crop[..., 2] = 255  # red in BGR
    x = arc_preprocess([crop])
    assert x.shape == (1, 3, 112, 112)
    assert np.allclose(x[0, 0], 1.0) and np.allclose(x[0, 2], -1.0)  # RGB order, [-1, 1]


def test_arcface_wrapper_normalises(tmp_path):
    w = np.random.default_rng(0).normal(size=(3 * 112 * 112, 512)) * 0.01
    path = linear_onnx(tmp_path / "arc.onnx", [1, 3, 112, 112], w, np.zeros(512))
    m = ArcFaceMatcher(path, "test")
    s = face_sample()
    emb = m.embed(s.best.arcface112)
    assert emb.shape == (512,) and abs(np.linalg.norm(emb) - 1) < 1e-5
    s.template, s.probe_embedding = emb, emb
    assert m.score(s) == pytest.approx(1.0, abs=1e-5)


def test_matcher_stub_and_missing_template():
    s = face_sample()
    assert MatcherStub().score(s) == -1.0
    s.template = s.probe_embedding = MatcherStub().embed(None)
    assert MatcherStub().score(s) == pytest.approx(1.0)
    assert cosine(np.array([1.0, 0]), np.array([0, 2.0])) == pytest.approx(0.0)


# ---- M1 -------------------------------------------------------------------------


def test_minifasnet_real_class_and_averaging(tmp_path):
    # Logits independent of input: model 1 says P(real) high, model 2 says it low.
    zero = np.zeros((3 * 80 * 80, 3))
    p1 = linear_onnx(tmp_path / "a.onnx", [1, 3, 80, 80], zero, np.array([0.0, 5.0, 0.0]))
    p2 = linear_onnx(tmp_path / "b.onnx", [1, 3, 80, 80], zero, np.array([5.0, 0.0, 0.0]))
    m1 = MiniFASNetPAD([p1, p2], "test")
    p_real_1 = softmax(np.array([0.0, 5.0, 0.0]))[1]
    p_real_2 = softmax(np.array([5.0, 0.0, 0.0]))[1]
    assert m1.score(face_sample()) == pytest.approx(1 - (p_real_1 + p_real_2) / 2, abs=1e-5)
    assert m1.score(Sample(jpegs=[], metadata={}, actions=[])) == 0.0


def test_minifasnet_uses_raw_0_255_bgr(tmp_path):
    # Logit for "real" = mean of channel 0 (blue in BGR), unnormalised.
    w = np.zeros((3 * 80 * 80, 3))
    w[: 80 * 80, 1] = 1.0 / (80 * 80)
    m1 = MiniFASNetPAD([linear_onnx(tmp_path / "c.onnx", [1, 3, 80, 80], w, np.zeros(3))], "t")
    blue = np.zeros((80, 80, 3), np.uint8)
    blue[..., 0] = 10
    assert m1.p_real([[blue]])[0] == pytest.approx(softmax(np.array([0.0, 10.0, 0.0]))[1], abs=1e-4)


# ---- M2 -------------------------------------------------------------------------


def test_dmad_trains_and_scores(tmp_path):
    rng = np.random.default_rng(0)
    refs = rng.normal(size=(200, 16))
    genuine = refs + rng.normal(scale=0.1, size=refs.shape)
    morph_refs = refs + rng.normal(scale=1.0, size=refs.shape)  # references far from the live probe
    x = np.vstack([[dmad_feature(r, p) for r, p in zip(refs, genuine)], [dmad_feature(m, p) for m, p in zip(morph_refs, genuine)]])
    y = np.array([0] * 200 + [1] * 200)
    save_m2(train_svm(x, y), tmp_path / "m2.joblib", "test-v1")
    det = DMADMorph(tmp_path / "m2.joblib")
    assert det.version == "test-v1"
    assert det.score_pair(refs[0], genuine[0]) < 0.5 < det.score_pair(morph_refs[1], genuine[1])
    s = Sample(jpegs=[], metadata={}, actions=[], template=refs[0], probe_embedding=genuine[0])
    assert det.score(s) < 0.5


# ---- M3 -------------------------------------------------------------------------


def test_gend_preprocess_matches_clip():
    crop = np.full((224, 224, 3), 255, np.uint8)
    x = gend_preprocess([crop])
    assert x.shape == (1, 3, 224, 224)
    assert np.allclose(x[0, :, 0, 0], (1.0 - CLIP_MEAN) / CLIP_STD, atol=1e-5)


def test_gend_fake_index(tmp_path):
    zero = np.zeros((3 * 224 * 224, 2))
    path = linear_onnx(tmp_path / "g.onnx", [1, 3, 224, 224], zero, np.array([0.0, 3.0]))
    assert GenDDeepfake(path, "t", fake_index=1).score(face_sample(2)) == pytest.approx(softmax(np.array([0.0, 3.0]))[1], abs=1e-5)
    assert GenDDeepfake(path, "t", fake_index=0).score(face_sample(2)) == pytest.approx(softmax(np.array([0.0, 3.0]))[0], abs=1e-5)


# ---- M4 -------------------------------------------------------------------------


def signals(spec: list[tuple[float, float | None, float | None]]) -> list[FrameSignal]:
    return [FrameSignal(t, b, y) for t, b, y in spec]


def test_blink_events_need_close_then_open():
    s = signals([(0, 0.1, 0), (100, 0.8, 0), (200, 0.1, 0), (300, 0.6, 0), (400, 0.4, 0), (500, 0.2, 0)])
    assert [e.kind for e in detect_events(s, 0.5, 0.3, 15)] == ["blink", "blink"]


def test_turn_events_reset_at_neutral():
    s = signals([(0, None, 0), (100, None, 20), (200, None, 25), (300, None, 3), (400, None, -20)])
    assert [(e.kind, e.t_ms) for e in detect_events(s, 0.5, 0.3, 15)] == [("turn_left", 100), ("turn_right", 400)]


def test_actions_must_be_in_order():
    events = [Event("blink", 100), Event("blink", 300), Event("turn_left", 500)]
    assert actions_in_order(events, ["blink_twice", "turn_left"])
    assert not actions_in_order(events, ["turn_left", "blink_twice"])
    assert not actions_in_order(events, ["blink_twice", "turn_right"])
    assert actions_in_order([Event("turn_right", 50), Event("turn_left", 60)], ["turn_right", "turn_left"])


def test_yaw_from_matrix():
    a = np.radians(30)
    m = np.eye(4)
    m[:3, :3] = [[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]]
    assert yaw_from_matrix(m) == pytest.approx(30)


def test_frame_times_and_subsample():
    meta = synthetic_metadata(n=6)
    times = frame_times_ms(meta, 6)
    assert times[0] == 0 and len(times) == 6
    assert frame_times_ms({}, 3) == pytest.approx([0, 1000 / 30, 2000 / 30])
    assert subsample([i * 1000 / 30 for i in range(30)], 15) == list(range(0, 30, 2))


# ---- M5 -------------------------------------------------------------------------


def test_virtual_name_rule():
    names = ["obs virtual", "v4l2loopback"]
    assert is_virtual_name("OBS Virtual Camera", names)
    assert not is_virtual_name("Integrated Webcam", names) and not is_virtual_name(None, names)


def test_features_and_rules_only_detector():
    jpg = encode_jpeg(np.zeros((8, 8, 3), np.uint8))
    meta = synthetic_metadata(n=4)
    f = extract(meta, [jpg] * 4, ["obs virtual"])
    assert set(f) == set(FEATURE_NAMES)
    assert f["dup_frame_ratio"] == 1.0 and f["virtual_name"] == 0.0 and f["res_mismatch"] == 0.0
    assert to_vector(f).shape == (len(FEATURE_NAMES),)
    det = InjectionDetector(["obs virtual"], "m5")
    assert det.version == "m5-rules-only"
    assert det.score(Sample(jpegs=[jpg] * 4, metadata=meta, actions=[])) == 0.0
    assert det.score(Sample(jpegs=[jpg] * 4, metadata={**meta, "device_name": "OBS Virtual Camera"}, actions=[])) == 1.0


def test_m5_classifier_round_trip(tmp_path):
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, len(FEATURE_NAMES)))
    y = (x[:, FEATURE_NAMES.index("dt_cv")] > 0).astype(int)
    save_m5(train_m5(x, y), tmp_path / "m5.joblib", "m5-test")
    det = InjectionDetector(["obs virtual"], "m5", tmp_path / "m5.joblib")
    assert det.version == "m5-test"
    feats = dict(zip(FEATURE_NAMES, np.zeros(len(FEATURE_NAMES))))
    assert 0.0 <= det.score_features({**feats, "dt_cv": 2.0}) <= 1.0
    assert det.score_features({**feats, "dt_cv": 2.0}) > det.score_features({**feats, "dt_cv": -2.0})


# ---- decision -----------------------------------------------------------------------

THRESHOLDS = {"match": 0.4, "m1_pad": 0.5, "m2_morph": 0.5, "m3_deepfake": 0.5, "m4_replay": 0.5, "m5_injection": 0.5}
CLEAN = {"match": 0.8, "m1_pad": 0.1, "m2_morph": 0.1, "m3_deepfake": 0.1, "m4_replay": 0.0, "m5_injection": 0.0}


def test_decide_accepts_clean_and_rejects_low_match():
    assert decide(CLEAN, THRESHOLDS) == (True, None)
    assert decide({**CLEAN, "match": 0.39}, THRESHOLDS) == (False, Reason.NO_MATCH)


def test_decide_priority_injection_first():
    scores = {**CLEAN, "m1_pad": 0.9, "m5_injection": 0.9, "match": 0.0}
    assert decide(scores, THRESHOLDS) == (False, Reason.INJECTION_FAIL)


def test_decide_threshold_is_strict_greater():
    assert decide({**CLEAN, "m1_pad": 0.5}, THRESHOLDS) == (True, None)


def test_stub_detector():
    s = Stub("m1_pad", 0.25)
    assert s.version == "stub" and s.score(Sample(jpegs=[], metadata={}, actions=[])) == 0.25
