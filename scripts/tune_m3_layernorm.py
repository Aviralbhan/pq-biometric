"""S7: LayerNorm-only tuning of GenD (M3) on OpenForensics Val, on the GPU.

    # in the GenD venv (transformers 4.56; GenD's own loader does not run on 5.x):
    .venv-gend456/Scripts/python scripts/tune_m3_layernorm.py split
    .venv-gend456/Scripts/python scripts/tune_m3_layernorm.py tune [--epochs 2] [--batch 16]
    .venv-gend456/Scripts/python scripts/export_onnx.py m3 --hf-dir models/weights/m3_deepfake/GenD_CLIP_L_14_ln \
        --out models/weights/m3_deepfake/gend_clip_l14_ln.onnx --images data/splits/m3_val_calib.csv --skip-sample-check

Protocol (CLAUDE.md, M3):
- Val is split 80/20 *by source image* (seed 0), so faces from one scene never sit on both
  sides: data/splits/m3_val_tune.csv (split "tune") and m3_val_calib.csv (split "calib").
- Only LayerNorm weights and biases of the CLIP vision encoder are trained; the backbone's
  other weights and GenD's linear head stay frozen. Class-weighted cross-entropy (Val has
  ~2.5 fakes per real face), AdamW, fp16 autocast, random horizontal flip.
- If a batch of 4 does not fit in GPU memory the run stops and says so (then M3 stays zero-shot).
- After each epoch: AUC / EER on the calib split (GPU, TF32 off). The last epoch is saved with
  save_pretrained to models/weights/m3_deepfake/GenD_CLIP_L_14_ln and exported to ONNX separately.

Writes eval/results/m3_ln_tuning.csv (per-epoch loss and calib AUC/EER).
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GEND_REPO = ROOT / "third_party" / "GenD"
BASE = ROOT / "models" / "weights" / "m3_deepfake" / "GenD_CLIP_L_14"
OUT = ROOT / "models" / "weights" / "m3_deepfake" / "GenD_CLIP_L_14_ln"
SPLITS = ROOT / "data" / "splits"
RESULTS = ROOT / "eval" / "results"


def split(seed: int = 0, tune_fraction: float = 0.8) -> None:
    with open(SPLITS / "m3_val.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    images = sorted({r["source_image"] for r in rows})
    random.Random(seed).shuffle(images)
    tune_images = set(images[: int(len(images) * tune_fraction)])
    for name, part, keep in (("m3_val_tune.csv", "tune", True), ("m3_val_calib.csv", "calib", False)):
        sel = [{**r, "split": part} for r in rows if (r["source_image"] in tune_images) == keep]
        with open(SPLITS / name, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(sel[0]))
            w.writeheader()
            w.writerows(sel)
        real = sum(r["label"] == "bona_fide" for r in sel)
        print(f"{name}: {len({r['source_image'] for r in sel})} source images, {real} real / {len(sel) - real} fake faces")
    assert not ({r["source_image"] for r in rows if r["source_image"] in tune_images} & (set(images) - tune_images))


def tune(epochs: int, batch: int, lr: float, seed: int) -> int:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # everything is already in the local Hugging Face cache
    import numpy as np
    import torch
    from PIL import Image

    sys.path.insert(0, str(GEND_REPO))
    from src.hf.modeling_gend import GenD

    sys.path.insert(0, str(ROOT))
    from eval.metrics import auc, eer

    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    model = GenD.from_pretrained(str(BASE)).to(device)
    for p in model.parameters():
        p.requires_grad = False
    ln_params = []
    for name, module in model.feature_extractor.vision_model.named_modules():
        if isinstance(module, torch.nn.LayerNorm):
            for p in module.parameters():
                p.requires_grad = True
                ln_params.append(p)
    n_ln = sum(p.numel() for p in ln_params)
    print(f"trainable LayerNorm parameters: {n_ln:,} of {sum(p.numel() for p in model.parameters()):,}")

    def load(name: str) -> list[dict]:
        with open(SPLITS / name, encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    tune_rows, calib_rows = load("m3_val_tune.csv"), load("m3_val_calib.csv")
    preprocess = model.feature_extractor.preprocess

    def tensor(path: str, flip: bool = False) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        if flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        return preprocess(img)

    n_real = sum(r["label"] == "bona_fide" for r in tune_rows)
    weights = torch.tensor([len(tune_rows) / (2 * n_real), len(tune_rows) / (2 * (len(tune_rows) - n_real))], device=device)
    opt = torch.optim.AdamW(ln_params, lr=lr, weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda")
    rng = random.Random(seed)

    def evaluate() -> tuple[float, float]:
        model.eval()
        probs, labels = [], []
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            for i in range(0, len(calib_rows), 32):
                chunk = calib_rows[i : i + 32]
                x = torch.stack([tensor(r["path"]) for r in chunk]).to(device)
                probs += model(x).float().softmax(-1)[:, 1].cpu().tolist()
                labels += [r["label"] == "attack" for r in chunk]
        p, y = np.array(probs), np.array(labels)
        return auc(p[y], p[~y]), eer(p[y], p[~y])[0]

    log = []
    base_auc, base_eer = evaluate()
    log.append({"epoch": 0, "train_loss": "", "calib_auc": base_auc, "calib_eer": base_eer, "batch": batch, "lr": lr})
    print(f"epoch 0 (zero-shot, fp16 eval): calib AUC {base_auc:.4f}  EER {base_eer:.4f}", flush=True)

    while True:  # find a batch size that fits
        try:
            model.train()
            x = torch.stack([tensor(r["path"]) for r in tune_rows[:batch]]).to(device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), torch.zeros(batch, dtype=torch.long, device=device), weight=weights)
            scaler.scale(loss).backward()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            print(f"batch {batch} fits (peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB)", flush=True)
            break
        except torch.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if batch <= 4:
                print("does not fit in GPU memory even at batch 4 with fp16; stopping (M3 stays zero-shot)")
                return 2
            batch //= 2

    for epoch in range(1, epochs + 1):
        model.train()
        order = list(range(len(tune_rows)))
        rng.shuffle(order)
        t0, losses = time.perf_counter(), []
        for step, i in enumerate(range(0, len(order), batch)):
            chunk = [tune_rows[j] for j in order[i : i + batch]]
            x = torch.stack([tensor(r["path"], flip=rng.random() < 0.5) for r in chunk]).to(device)
            y = torch.tensor([int(r["label"] == "attack") for r in chunk], device=device)
            with torch.autocast("cuda", dtype=torch.float16):
                loss = torch.nn.functional.cross_entropy(model(x).float(), y, weight=weights)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
            if step % 100 == 0:
                print(f"  epoch {epoch} step {step}/{len(order) // batch}: loss {np.mean(losses[-100:]):.4f}", flush=True)
        a, e = evaluate()
        log.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "calib_auc": a, "calib_eer": e, "batch": batch, "lr": lr})
        print(f"epoch {epoch}: loss {np.mean(losses):.4f}  calib AUC {a:.4f}  EER {e:.4f}  ({time.perf_counter() - t0:.0f} s)", flush=True)

    model.save_pretrained(str(OUT))
    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "m3_ln_tuning.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(log[0]))
        w.writeheader()
        w.writerows(log)
    print(f"saved {OUT}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["split", "tune"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if args.command == "split":
        split(args.seed)
        return 0
    return tune(args.epochs, args.batch, args.lr, args.seed)


if __name__ == "__main__":
    raise SystemExit(main())
