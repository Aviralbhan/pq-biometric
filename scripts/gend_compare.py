"""Score images with GenD exactly as its own code does, to compare transformers versions.

    python scripts/gend_compare.py --images list.txt --out var/gend_5.19.json        # main venv (transformers 5.x)
    .venv-gend456/Scripts/python scripts/gend_compare.py --images list.txt --out var/gend_4.56.json
    python scripts/gend_compare.py --backend onnx --images list.txt --out var/gend_onnx.json   # the file the server runs
    python scripts/gend_compare.py --compare var/gend_4.56.json var/gend_onnx.json --csv eval/results/m3_gend_equivalence.csv

The torch backend uses only torch, transformers and PIL, so it runs in the small 4.56 venv. Loading
follows GenD's README: GenD.from_pretrained(...), model.feature_extractor.preprocess(img),
softmax over the logits. The first run downloads openai/clip-vit-large-patch14 into the
Hugging Face cache (GenD's loader builds its backbone from it).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEND_REPO = ROOT / "third_party" / "GenD"
GEND_WEIGHTS = ROOT / "models" / "weights" / "m3_deepfake" / "GenD_CLIP_L_14"


def score_onnx(images: list[str]) -> dict:
    """The deployed path: server preprocessing (detectors.m3_deepfake.gend) + the exported ONNX file, CPU."""
    import cv2
    import onnxruntime

    sys.path.insert(0, str(ROOT))
    from detectors.base import onnx_session, softmax
    from detectors.m3_deepfake.gend import preprocess

    sess = onnx_session(GEND_WEIGHTS.parent / "gend_clip_l14.onnx", threads=8)
    probs = {}
    for path in images:
        x = preprocess([cv2.resize(cv2.imread(path), (224, 224))])
        probs[path] = softmax(sess.run(None, {"pixel_values": x})[0])[0].tolist()
    return {"backend": f"onnxruntime {onnxruntime.__version__} + server preprocessing", "device": "cpu", "probs": probs}


def score(images: list[str], device: str) -> dict:
    import torch
    import transformers
    from PIL import Image

    sys.path.insert(0, str(GEND_REPO))
    from src.hf.modeling_gend import GenD

    label = f"GenD own pipeline, transformers {transformers.__version__}"
    try:
        model = GenD.from_pretrained(str(GEND_WEIGHTS)).to(device).eval()
    except Exception as exc:  # recorded, not hidden: e.g. transformers 5 refuses GenD's loader
        return {"backend": label, "device": device, "error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}", "probs": {}}
    probs = {}
    with torch.no_grad():
        for path in images:
            x = model.feature_extractor.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            probs[path] = model(x).softmax(dim=-1)[0].cpu().tolist()
    return {"backend": label, "torch": torch.__version__, "device": device, "probs": probs}


def compare(a_path: str, b_path: str, tol: float, csv_path: str | None = None) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    name = f"{a['backend']} ({a['device']}) vs {b['backend']} ({b['device']})"
    row = {"comparison": name, "n": 0, "max_abs_dp": "", "mean_abs_dp": "", "same_class": "", "result": ""}
    if a.get("error") or b.get("error"):
        row["result"] = "could not run: " + (a.get("error") or b.get("error"))
        ok = False
    else:
        common = sorted(set(a["probs"]) & set(b["probs"]))
        diffs = [max(abs(x - y) for x, y in zip(a["probs"][k], b["probs"][k])) for k in common]
        same = sum((a["probs"][k][1] > 0.5) == (b["probs"][k][1] > 0.5) for k in common)
        ok = max(diffs) <= tol and same == len(common)
        row.update(n=len(common), max_abs_dp=f"{max(diffs):.3g}", mean_abs_dp=f"{sum(diffs) / len(diffs):.3g}",
                   same_class=f"{same}/{len(common)}", result=("match" if ok else "differs") + f" (tolerance {tol})")
    print(f"{name}: " + ", ".join(f"{k}={v}" for k, v in row.items() if k != "comparison"))
    if csv_path:
        import csv

        new = not Path(csv_path).exists()
        with open(csv_path, "a", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                w.writeheader()
            w.writerow(row)
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="text file, one image path per line")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--backend", choices=["torch", "onnx"], default="torch")
    ap.add_argument("--csv", help="append the comparison to this CSV")
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare, args.tol, args.csv)
    images = [line.strip() for line in Path(args.images).read_text().splitlines() if line.strip()]
    result = score_onnx(images) if args.backend == "onnx" else score(images, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"{result['backend']} ({result['device']}): {len(result['probs'])} images -> {args.out}" + (f"  ERROR {result['error']}" if result.get("error") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
