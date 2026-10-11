"""S7: fine-tune the pretrained MiniFASNet models (M1) on CelebA-Spoof.

    pip install -r requirements-export.txt
    python scripts/fetch_models.py yunet minifasnet
    python scripts/filter_celeba_spoof.py
    python scripts/train_m1.py [--epochs 5] [--limit 50000]
    python scripts/export_onnx.py m1 --checkpoints models/weights/m1_pad/finetuned

Patches are cut with the server's own face preprocessing (YuNet box -> Silent-Face
crop at each model's scale), cached once in var/m1_cache/, then each model is
fine-tuned with cross-entropy on 3 classes: 1 = live (Silent-Face's "real" class),
0 = print/paper/cloth (photo, poster, A4, masks), 2 = screen replay and 3D mask.
The server only uses P(class 1). The checkpoint with the lowest val EER is kept.
Writes eval/results/m1_finetune.csv.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from eval.harness import write_csv  # noqa: E402
from eval.metrics import eer  # noqa: E402
from preprocessing.faces import FacePreprocessor  # noqa: E402
from settings import load_config, resolve  # noqa: E402

SILENT_FACE = ROOT / "third_party" / "Silent-Face-Anti-Spoofing"
CLASS_OF = {"": 1, "photo": 0, "poster": 0, "a4": 0, "face_mask": 0, "upper_body_mask": 0, "region_mask": 0,
            "pc": 2, "pad": 2, "phone": 2, "3d_mask": 2}


def build_cache(csv_path: Path, pre: FacePreprocessor, cache: Path, limit: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(patches[N, n_scales, 80, 80, 3] uint8, labels[N], is_attack[N])."""
    if cache.exists():
        d = np.load(cache)
        return d["x"], d["y"], d["attack"]
    with open(csv_path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))[:limit]
    xs, ys, attack = [], [], []
    for i, r in enumerate(rows):
        img = cv2.imread(r["path"])
        c = pre.process(img) if img is not None else None
        if c is None:
            continue
        xs.append(np.stack(c.pad80))
        ys.append(CLASS_OF[r["attack_type"]])
        attack.append(r["label"] == "attack")
        if i % 5000 == 0:
            print(f"  {csv_path.name}: {i}/{len(rows)}")
    x, y, a = np.stack(xs), np.array(ys), np.array(attack)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, x=x, y=y, attack=a)
    return x, y, a


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--limit", type=int, help="max training images (the full train split is ~400k)")
    ap.add_argument("--splits", default=str(ROOT / "data" / "splits"))
    ap.add_argument("--out", default=str(ROOT / "models" / "weights" / "m1_pad" / "finetuned"))
    ap.add_argument("--config")
    args = ap.parse_args(argv)

    import torch
    import torch.nn.functional as F

    sys.path.insert(0, str(SILENT_FACE))
    from src.anti_spoof_predict import MODEL_MAPPING
    from src.utility import get_kernel, parse_model_name

    config = load_config(args.config, {"detectors": {"allow_stubs": False}})
    pre = FacePreprocessor.from_config(config)
    cache = ROOT / "var" / "m1_cache"
    x_tr, y_tr, _ = build_cache(Path(args.splits) / "celeba_spoof_train.csv", pre, cache / "train.npz", args.limit)
    x_va, _, a_va = build_cache(Path(args.splits) / "celeba_spoof_val.csv", pre, cache / "val.npz", None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for scale_idx, target in enumerate(resolve(p) for p in config["detectors"]["m1_pad"]["models"]):
        name = target.stem + ".pth"
        h, w, model_type, _ = parse_model_name(name)
        model = MODEL_MAPPING[model_type](conv6_kernel=get_kernel(h, w)).to(device)
        state = torch.load(SILENT_FACE / "resources" / "anti_spoof_models" / name, map_location=device)
        model.load_state_dict({k.removeprefix("module."): v for k, v in state.items()})
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

        def to_tensor(batch: np.ndarray) -> torch.Tensor:
            return torch.from_numpy(batch.transpose(0, 3, 1, 2).astype(np.float32)).to(device)  # raw 0-255 BGR, as Silent-Face

        def val_eer() -> float:
            model.eval()
            p_real = []
            with torch.no_grad():
                for i in range(0, len(x_va), 1024):
                    p_real.append(F.softmax(model(to_tensor(x_va[i : i + 1024, scale_idx])), 1)[:, 1].cpu().numpy())
            spoof = 1 - np.concatenate(p_real)
            return eer(spoof[a_va], spoof[~a_va])[0]

        best = val_eer()
        rows.append({"model": name, "epoch": 0, "val_eer": best})
        print(f"{name}: pretrained val EER {best:.4f}")
        torch.save(model.state_dict(), out_dir / name)
        rng = np.random.default_rng(0)
        for epoch in range(1, args.epochs + 1):
            model.train()
            order = rng.permutation(len(x_tr))
            for i in range(0, len(order), args.batch):
                idx = order[i : i + args.batch]
                batch = x_tr[idx, scale_idx]
                flip = rng.random(len(idx)) < 0.5
                batch[flip] = batch[flip, :, ::-1]
                logits = model(to_tensor(batch))
                logits = logits[0] if isinstance(logits, tuple) else logits
                loss = F.cross_entropy(logits, torch.from_numpy(y_tr[idx]).to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            e = val_eer()
            rows.append({"model": name, "epoch": epoch, "val_eer": e})
            print(f"  epoch {epoch}: val EER {e:.4f}")
            if e < best:
                best = e
                torch.save(model.state_dict(), out_dir / name)
    write_csv(ROOT / "eval" / "results" / "m1_finetune.csv", rows)
    print(f"best checkpoints in {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
