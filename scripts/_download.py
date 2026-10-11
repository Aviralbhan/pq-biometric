"""Download helpers: streamed download with size and optional MD5 check, and archive extraction."""

from __future__ import annotations

import hashlib
import shutil
import tarfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path


def download(
    url: str, dest: Path, expected_size: int | None = None, md5: str | None = None, chunk: int = 1 << 20, retries: int = 50
) -> Path:
    """Stream url to dest (skipped if dest already has the expected size). Prints progress every 5%.

    Dropped connections are resumed from the partial file with an HTTP Range request.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (expected_size is None or dest.stat().st_size == expected_size) and (md5 is None or file_md5(dest) == md5):
        print(f"  have {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(retries + 1):
        have = tmp.stat().st_size if tmp.exists() else 0
        if expected_size is not None and have == expected_size:
            break
        headers = {"User-Agent": "pq-biometric/1.0"}
        if have:
            headers["Range"] = f"bytes={have}-"
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                if have and resp.status != 206:  # server ignored the range: start over
                    have = 0
                total = expected_size or have + int(resp.headers.get("Content-Length") or 0)
                done, next_report = have, (have / total if total else 0) + 0.05
                with open(tmp, "ab" if have else "wb") as out:
                    while block := resp.read(chunk):
                        out.write(block)
                        done += len(block)
                        if total and done / total >= next_report:
                            print(f"  {dest.name}: {100 * done / total:.0f}% of {total / 2**20:.0f} MiB", flush=True)
                            next_report += 0.05
            if expected_size is None or tmp.stat().st_size == expected_size:
                break
        except (OSError, urllib.error.URLError) as exc:
            if attempt == retries:
                raise
            print(f"  {dest.name}: connection dropped at {tmp.stat().st_size if tmp.exists() else 0} bytes ({exc}); resuming", flush=True)
            time.sleep(min(30, 2 + attempt))
    if expected_size is not None and tmp.stat().st_size != expected_size:
        raise IOError(f"{dest.name}: got {tmp.stat().st_size} bytes, expected {expected_size}")
    if md5 is not None and file_md5(tmp) != md5:
        raise IOError(f"{dest.name}: MD5 mismatch")
    tmp.replace(dest)
    return dest


def file_md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def extract(archive: Path, dest: Path) -> Path:
    """Extract zip or tar(.gz) into dest, refusing members that would escape it."""
    dest.mkdir(parents=True, exist_ok=True)
    root = dest.resolve()
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as zf:
            for name in zf.namelist():
                if not (root / name).resolve().is_relative_to(root):
                    raise IOError(f"unsafe path in {archive.name}: {name}")
            zf.extractall(dest)
    else:
        with tarfile.open(archive) as tf:
            tf.extractall(dest, filter="data")
    return dest


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()
