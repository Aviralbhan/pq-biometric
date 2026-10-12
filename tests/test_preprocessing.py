from __future__ import annotations

import numpy as np
import pytest

from preprocessing.faces import ARCFACE_TEMPLATE, CenterFaceStub, Face, FacePreprocessor, align_arcface, crop_scaled, crop_square
from preprocessing.video import decode_frames, encode_frames, pack_jpegs, sample_indices, unpack_jpegs
from tests.conftest import synthetic_frames


def test_video_round_trip():
    yy, xx = np.mgrid[0:120, 0:160]
    smooth = np.stack([xx, yy, xx + yy], axis=-1).astype(np.uint8)
    video = encode_frames([smooth] * 4 + synthetic_frames(1, h=120, w=160), quality=95)
    decoded = decode_frames(video)
    assert len(decoded) == 5 and decoded[0].shape == (120, 160, 3)
    assert np.abs(decoded[0].astype(int) - smooth.astype(int)).mean() < 3  # lossy, but the same image
    assert decode_frames(video, [4])[0].shape == (120, 160, 3)


def test_video_rejects_corruption():
    video = encode_frames(synthetic_frames(2, h=32, w=32))
    with pytest.raises(ValueError):
        unpack_jpegs(b"XXXXX" + video[5:])
    with pytest.raises(ValueError):
        unpack_jpegs(video[:-3])
    with pytest.raises(ValueError):
        unpack_jpegs(video + b"\x00")


def test_pack_empty():
    assert unpack_jpegs(pack_jpegs([])) == []


@pytest.mark.parametrize("n, k, expected", [(10, 3, [0, 4, 9]), (3, 8, [0, 1, 2]), (0, 4, []), (5, 1, [2])])
def test_sample_indices(n, k, expected):
    assert sample_indices(n, k) == expected


def test_align_arcface_identity_on_template():
    img = np.random.default_rng(0).integers(0, 255, (112, 112, 3), dtype=np.uint8)
    out = align_arcface(img, ARCFACE_TEMPLATE.copy(), 112)
    assert out.shape == (112, 112, 3)
    assert np.abs(out.astype(int) - img.astype(int)).mean() < 2


def test_align_arcface_undoes_scale_and_shift():
    img = np.zeros((400, 400, 3), np.uint8)
    lm = ARCFACE_TEMPLATE * 2 + 50  # face drawn at twice the size, offset by 50 px
    for x, y in lm.astype(int):
        img[y - 3 : y + 4, x - 3 : x + 4] = 255
    out = align_arcface(img, lm, 112)
    for x, y in ARCFACE_TEMPLATE.astype(int):
        assert out[y, x].max() > 100


def test_crop_scaled_matches_silent_face_shift_rule():
    img = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
    out = crop_scaled(img, (0, 0, 20, 20), 2.7, 80, 80)  # box at the corner: crop is shifted inside
    assert out.shape == (80, 80, 3)
    big = crop_scaled(img, (10, 10, 80, 80), 4.0, 80, 80)  # scale capped by image size
    assert big.shape == (80, 80, 3)


def test_crop_square_pads_outside_image():
    img = np.full((50, 50, 3), 200, np.uint8)
    out = crop_square(img, (-10, -10, 30, 30), 1.0, 224)
    assert out.shape == (224, 224, 3)
    assert out[0, 0].sum() == 0 and out[-1, -1].sum() > 0


class FixedDetector:
    version = "fixed"

    def __init__(self, faces):
        self.faces = faces

    def detect(self, img):
        return self.faces


def test_preprocessor_filters_small_faces():
    small = Face((0, 0, 50, 50), ARCFACE_TEMPLATE / 112 * 50, 0.99)
    pre = FacePreprocessor(FixedDetector([small]), min_face_px=112)
    assert pre.process(np.zeros((480, 640, 3), np.uint8)) is None


def test_face_choice_handles_both_frll_failure_cases():
    real = Face((100, 100, 200, 240), ARCFACE_TEMPLATE / 112 * 200 + 100, 0.88)
    weak_big = Face((90, 330, 260, 300), ARCFACE_TEMPLATE / 112 * 260 + 90, 0.65)  # subject 030
    pre = FacePreprocessor(FixedDetector([real, weak_big]), min_face_px=112)
    assert pre.best_face(np.zeros((700, 700, 3), np.uint8)) is real
    small_confident = Face((358, 519, 200, 249), ARCFACE_TEMPLATE / 112 * 200 + 358, 0.837)  # subject 114
    real_3q = Face((550, 375, 445, 566), ARCFACE_TEMPLATE / 112 * 445 + 550, 0.811)
    pre = FacePreprocessor(FixedDetector([small_confident, real_3q]), min_face_px=112)
    assert pre.best_face(np.zeros((1350, 1350, 3), np.uint8)) is real_3q


def test_small_real_face_is_not_replaced_by_false_box():
    real_small = Face((300, 200, 105, 140), ARCFACE_TEMPLATE / 112 * 105 + 300, 0.93)
    false_big = Face((0, 300, 200, 180), ARCFACE_TEMPLATE / 112 * 200, 0.55)
    pre = FacePreprocessor(FixedDetector([real_small, false_big]), min_face_px=112)
    assert pre.best_face(np.zeros((480, 640, 3), np.uint8)) is None


def test_preprocessor_produces_all_crops():
    pre = FacePreprocessor(CenterFaceStub(), min_face_px=112, pad_scales=(2.7, 4.0))
    crops = pre.process(np.zeros((480, 640, 3), np.uint8))
    assert crops.arcface112.shape == (112, 112, 3)
    assert crops.gend224.shape == (224, 224, 3)
    assert [p.shape for p in crops.pad80] == [(80, 80, 3), (80, 80, 3)]
    assert pre.version == "stub"
