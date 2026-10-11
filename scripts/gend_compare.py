"""Score images with GenD exactly as its own code does, to compare transformers versions.

    python scripts/gend_compare.py --images list.txt --out var/gend_5.19.json        # main venv (transformers 5.x)
    .venv-gend456/Scripts/python scripts/gend_compare.py --images list.txt --out var/gend_4.56.json
    python scripts/gend_compare.py --compare var/gend_5.19.json var/gend_4.56.json

Uses only torch, transformers and PIL, so it runs in the small 4.56 venv. Loading
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


def score(images: list[str], device: str) -> dict:
    import torch
    import transformers
    from PIL import Image

    sys.path.insert(0, str(GEND_REPO))
    from src.hf.modeling_gend import GenD

    model = GenD.from_pretrained(str(GEND_WEIGHTS)).to(device).eval()
    probs = {}
    with torch.no_grad():
        for path in images:
            x = model.feature_extractor.preprocess(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
            probs[path] = model(x).softmax(dim=-1)[0].cpu().tolist()
    return {"transformers": transformers.__version__, "torch": torch.__version__, "device": device, "probs": probs}


def compare(a_path: str, b_path: str, tol: float) -> int:
    a, b = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    common = sorted(set(a["probs"]) & set(b["probs"]))
    diffs = [max(abs(x - y) for x, y in zip(a["probs"][k], b["probs"][k])) for k in common]
    argmax_same = sum(
        max(range(2), key=a["probs"][k].__getitem__) == max(range(2), key=b["probs"][k].__getitem__) for k in common
    )
    print(f"{a['transformers']} vs {b['transformers']} on {len(common)} images ({a['device']}):")
    print(f"  max |dP| = {max(diffs):.2e}, mean |dP| = {sum(diffs) / len(diffs):.2e}, same class {argmax_same}/{len(common)}")
    ok = max(diffs) <= tol and argmax_same == len(common)
    print(f"  {'MATCH' if ok else 'DIFFERENT'} (tolerance {tol})")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", help="text file, one image path per line")
    ap.add_argument("--out")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--tol", type=float, default=1e-4)
    args = ap.parse_args(argv)
    if args.compare:
        return compare(*args.compare, args.tol)
    images = [line.strip() for line in Path(args.images).read_text().splitlines() if line.strip()]
    result = score(images, args.device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"scored {len(images)} images with transformers {result['transformers']} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
