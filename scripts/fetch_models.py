"""Download pretrained model files into models/weights/ (gitignored; never committed).

    python scripts/fetch_models.py yunet landmarker      # small: ~0.2 MB + ~3.7 MB
    python scripts/fetch_models.py buffalo_l             # ArcFace, ~280 MB zip (InsightFace, non-commercial research)
    python scripts/fetch_models.py minifasnet            # Silent-Face-Anti-Spoofing code + 2 checkpoints (Apache-2.0) into third_party/
    python scripts/fetch_models.py gend                  # GenD CLIP-L/14 via Hugging Face (MIT), ~1.2 GB; needs requirements-export.txt
    python scripts/fetch_models.py all

MiniFASNet and GenD are PyTorch checkpoints; scripts/export_onnx.py turns them
into the ONNX files the server loads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts._download import download, extract, remove  # noqa: E402
from settings import ROOT  # noqa: E402

WEIGHTS = ROOT / "models" / "weights"
THIRD_PARTY = ROOT / "third_party"

YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
LANDMARKER_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
BUFFALO_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
SILENT_FACE_REPO_NAME = "minivision-ai/Silent-Face-Anti-Spoofing"
GEND_REPO = "https://github.com/yermandy/GenD.git"
GEND_HF = "yermandy/GenD_CLIP_L_14"


def yunet() -> None:
    download(YUNET_URL, WEIGHTS / "face_detection_yunet_2023mar.onnx")


def landmarker() -> None:
    download(LANDMARKER_URL, WEIGHTS / "face_landmarker.task")


def buffalo_l() -> None:
    target = WEIGHTS / "buffalo_l" / "w600k_r50.onnx"
    if target.exists():
        print("  have buffalo_l")
        return
    archive = download(BUFFALO_URL, WEIGHTS / "buffalo_l.zip")
    extract(archive, WEIGHTS / "buffalo_l")
    nested = WEIGHTS / "buffalo_l" / "buffalo_l"
    if nested.is_dir():  # some releases nest the folder
        for f in nested.iterdir():
            f.replace(WEIGHTS / "buffalo_l" / f.name)
        nested.rmdir()
    remove(archive)
    if not target.exists():
        raise FileNotFoundError(f"{target} not in the buffalo_l archive")


def clone(repo: str, dest: Path) -> None:
    if dest.exists():
        print(f"  have {dest.name}")
        return
    subprocess.run(["git", "clone", "--depth", "1", repo, str(dest)], check=True)


def fetch_github_files(repo: str, branch: str, prefixes: tuple[str, ...], dest: Path) -> str:
    """Download the files under `prefixes` at the branch's current commit, each checked against its git blob hash.

    Used instead of git clone when a repo has paths Windows cannot check out. Returns the commit sha.
    """
    def api(url: str) -> dict:
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json", "User-Agent": "pq-biometric"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.load(resp)

    commit = api(f"https://api.github.com/repos/{repo}/commits/{branch}")["sha"]
    tree = api(f"https://api.github.com/repos/{repo}/git/trees/{commit}?recursive=1")
    if tree.get("truncated"):
        raise RuntimeError(f"{repo}: tree listing truncated")
    for entry in tree["tree"]:
        if entry["type"] != "blob" or not entry["path"].startswith(prefixes):
            continue
        target = dest / entry["path"]
        if target.exists() and _git_blob_sha(target.read_bytes()) == entry["sha"]:
            continue
        with urllib.request.urlopen(f"https://raw.githubusercontent.com/{repo}/{commit}/{entry['path']}", timeout=120) as resp:
            data = resp.read()
        if _git_blob_sha(data) != entry["sha"]:
            raise IOError(f"{entry['path']}: content does not match the git blob hash")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
    (dest / "COMMIT").write_text(commit + "\n")
    return commit


def _git_blob_sha(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def minifasnet() -> None:
    # Not a git clone: the repo's saved_logs/ has a folder name ending in a space, which Windows
    # refuses to check out. Only the model code and the two pretrained checkpoints are needed.
    commit = fetch_github_files(SILENT_FACE_REPO_NAME, "master", ("src/", "resources/anti_spoof_models/"), THIRD_PARTY / "Silent-Face-Anti-Spoofing")
    print(f"  Silent-Face-Anti-Spoofing @ {commit[:12]}")


def gend() -> None:
    clone(GEND_REPO, THIRD_PARTY / "GenD")
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("pip install -r requirements-export.txt huggingface_hub first") from exc
    snapshot_download(GEND_HF, local_dir=str(WEIGHTS / "m3_deepfake" / "GenD_CLIP_L_14"))


TARGETS = {"yunet": yunet, "landmarker": landmarker, "buffalo_l": buffalo_l, "minifasnet": minifasnet, "gend": gend}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", choices=[*TARGETS, "all"])
    args = ap.parse_args(argv)
    for name in TARGETS if "all" in args.targets else args.targets:
        print(f"{name}:")
        TARGETS[name]()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
