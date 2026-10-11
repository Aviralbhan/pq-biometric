"""S10: export MiniFASNet (M1) and GenD (M3) to ONNX (opset 17) and check them against PyTorch.

    pip install -r requirements-export.txt
    python scripts/fetch_models.py minifasnet gend
    python scripts/export_onnx.py m1 [--checkpoints models/weights/m1_pad/finetuned]   # default: the pretrained .pth files
    python scripts/export_onnx.py m3 [--images data/splits/m3_val.csv]

Check: PyTorch vs ONNX output cosine similarity > 0.999 on 200 inputs (crops from
--images if given, otherwise random tensors), plus median CPU latency per input.
For M3 the fake class index is confirmed on GenD's own example images (one
FaceForensics deepfake, one real frame) and must equal detectors.m3_deepfake.fake_index.
Appends to eval/results/onnx_export.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from detectors.base import onnx_session, softmax  # noqa: E402
from detectors.m3_deepfake.gend import preprocess as gend_preprocess  # noqa: E402
from eval.harness import RESULTS  # noqa: E402
from preprocessing.faces import FacePreprocessor  # noqa: E402
from settings import load_config, resolve  # noqa: E402

SILENT_FACE = ROOT / "third_party" / "Silent-Face-Anti-Spoofing"
GEND_REPO = ROOT / "third_party" / "GenD"
GEND_SAMPLES = {
    "fake": "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/DF/000_003/000.png?raw=true",
    "real": "https://github.com/yermandy/deepfake-detection/blob/main/datasets/FF/real/000/000.png?raw=true",
}


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.reshape(len(a), -1), b.reshape(len(b), -1)
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12)


def check(torch_fn, onnx_path: Path, inputs: np.ndarray) -> dict:
    import torch

    sess = onnx_session(onnx_path, threads=2)
    name = sess.get_inputs()[0].name
    with torch.no_grad():
        ref = np.concatenate([torch_fn(torch.from_numpy(inputs[i : i + 16])).numpy() for i in range(0, len(inputs), 16)])
    out = np.concatenate([sess.run(None, {name: inputs[i : i + 16]})[0] for i in range(0, len(inputs), 16)])
    lat = []
    for i in range(min(50, len(inputs))):
        t = time.perf_counter_ns()
        sess.run(None, {name: inputs[i : i + 1]})
        lat.append((time.perf_counter_ns() - t) / 1e6)
    cos = cosine_rows(ref, out)
    return {"n": len(inputs), "cosine_min": float(cos.min()), "cosine_mean": float(cos.mean()),
            "max_abs_diff": float(np.abs(ref - out).max()), "cpu_ms_median": float(np.median(lat)), "pass": bool(cos.min() > 0.999)}


def image_inputs(csv_path: str | None, config: dict, kind: str, n: int = 200) -> np.ndarray | None:
    if not csv_path:
        return None
    with open(csv_path, encoding="utf-8") as fh:
        paths = [r["path"] for r in csv.DictReader(fh)]
    pre = FacePreprocessor.from_config(config)
    out = []
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            continue
        if kind == "m3":
            out.append(gend_preprocess([cv2.resize(img, (224, 224))])[0])
        else:
            c = pre.process(img)
            if c is not None:
                out.append(c.pad80)
        if len(out) >= n:
            break
    return np.stack(out).astype(np.float32) if out else None


def export_m1(args, config) -> list[dict]:
    import torch

    sys.path.insert(0, str(SILENT_FACE))
    from src.anti_spoof_predict import MODEL_MAPPING
    from src.utility import get_kernel, parse_model_name

    ckpt_dir = Path(args.checkpoints) if args.checkpoints else SILENT_FACE / "resources" / "anti_spoof_models"
    targets = [resolve(p) for p in config["detectors"]["m1_pad"]["models"]]
    rows = []
    patches = image_inputs(args.images, config, "m1")
    for target in targets:
        ckpt = ckpt_dir / (target.stem + ".pth")
        h, w, model_type, scale = parse_model_name(ckpt.name)
        model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(h, w))
        state = torch.load(ckpt, map_location="cpu")
        state = {k.removeprefix("module."): v for k, v in state.items()}
        model.load_state_dict(state)
        model.eval()
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.onnx.export(model, torch.zeros(1, 3, h, w), str(target), opset_version=17, input_names=["input"], output_names=["logits"],
                          dynamic_axes={"input": {0: "N"}, "logits": {0: "N"}}, dynamo=False)
        idx = targets.index(target)
        inputs = (patches[:, idx].transpose(0, 3, 1, 2).copy() if patches is not None
                  else np.random.default_rng(0).uniform(0, 255, (200, 3, h, w)).astype(np.float32))
        rows.append({"module": "m1_pad", "model": target.name, "source": str(ckpt), **check(model, target, inputs)})
    return rows


def export_m3(args, config) -> list[dict]:
    import torch

    sys.path.insert(0, str(GEND_REPO))
    from src.hf.modeling_gend import GenD

    model = GenD.from_pretrained(args.hf_dir)
    model.eval()
    target = resolve(config["detectors"]["m3_deepfake"]["model"])
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(model, torch.zeros(1, 3, 224, 224), str(target), opset_version=17, input_names=["pixel_values"],
                      output_names=["logits"], dynamic_axes={"pixel_values": {0: "N"}, "logits": {0: "N"}}, dynamo=False)

    samples = {}
    for label, url in GEND_SAMPLES.items():
        data = urllib.request.urlopen(url, timeout=60).read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        samples[label] = softmax(onnx_session(target).run(None, {"pixel_values": gend_preprocess([cv2.resize(img, (224, 224))])})[0])[0]
    fake_index = int(np.argmax(samples["fake"]))
    if int(np.argmax(samples["real"])) == fake_index:
        raise SystemExit(f"GenD gives the same class for its fake and real examples: {samples}")
    configured = int(config["detectors"]["m3_deepfake"].get("fake_index", 1))
    if fake_index != configured:
        raise SystemExit(f"GenD fake class is {fake_index}; set detectors.m3_deepfake.fake_index to {fake_index} in config.yaml")
    inputs = image_inputs(args.images, config, "m3")
    if inputs is None:
        inputs = np.random.default_rng(0).normal(0, 1, (200, 3, 224, 224)).astype(np.float32)
    return [{"module": "m3_deepfake", "model": target.name, "source": args.hf_dir, "fake_index": fake_index, **check(model, target, inputs)}]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("module", choices=["m1", "m3"])
    ap.add_argument("--checkpoints")
    ap.add_argument("--hf-dir", default=str(ROOT / "models" / "weights" / "m3_deepfake" / "GenD_CLIP_L_14"))
    ap.add_argument("--images", help="CSV with a path column; its faces are used for the PyTorch-vs-ONNX check")
    ap.add_argument("--config")
    args = ap.parse_args(argv)
    config = load_config(args.config)
    rows = export_m1(args, config) if args.module == "m1" else export_m3(args, config)
    out = RESULTS / "onnx_export.csv"
    new = not out.exists()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        fields = ["module", "model", "source", "fake_index", "n", "cosine_min", "cosine_mean", "max_abs_diff", "cpu_ms_median", "pass"]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(f"  {r['model']}: cosine min {r['cosine_min']:.6f}  CPU {r['cpu_ms_median']:.1f} ms  {'PASS' if r['pass'] else 'FAIL'}")
    return 0 if all(r["pass"] for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
