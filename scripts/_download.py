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


def download_parallel(url: str, dest: Path, size: int, md5: str | None = None, connections: int = 4, retries: int = 50) -> Path:
    """Download with `connections` parallel HTTP Range requests; resumable; MD5-checked at the end.

    Progress lives in <dest>.part (preallocated to `size`) and <dest>.part.json (bytes done per
    segment). An existing contiguous .part from download() is kept as an already-finished prefix.
    """
    import json
    import threading

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == size and (md5 is None or file_md5(dest) == md5):
        print(f"  have {dest.name}")
        return dest
    tmp = dest.with_suffix(dest.suffix + ".part")
    state_path = dest.with_suffix(dest.suffix + ".part.json")
    if state_path.exists():
        segments = json.loads(state_path.read_text())
    else:
        start = tmp.stat().st_size if tmp.exists() else 0  # contiguous prefix from a single-stream download
        bounds = [start + (size - start) * k // connections for k in range(connections + 1)]
        segments = [{"start": bounds[k], "end": bounds[k + 1], "pos": bounds[k]} for k in range(connections)]
        state_path.write_text(json.dumps(segments))  # before preallocating, so a crash cannot lose the prefix
    with open(tmp, "ab"):
        pass
    if tmp.stat().st_size < size:
        with open(tmp, "r+b") as fh:
            fh.truncate(size)
    lock = threading.Lock()
    errors: list[Exception] = []

    def save_state() -> None:
        state_path.write_text(json.dumps(segments))

    def worker(seg: dict) -> None:
        for attempt in range(retries + 1):
            if seg["pos"] >= seg["end"]:
                return
            headers = {"User-Agent": "pq-biometric/1.0", "Range": f"bytes={seg['pos']}-{seg['end'] - 1}"}
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as resp:
                    if resp.status != 206:
                        raise IOError(f"server ignored the Range header (HTTP {resp.status})")
                    with open(tmp, "r+b") as out:
                        out.seek(seg["pos"])
                        while seg["pos"] < seg["end"] and (block := resp.read(min(1 << 20, seg["end"] - seg["pos"]))):
                            out.write(block)
                            with lock:
                                seg["pos"] += len(block)
                                save_state()
            except (OSError, urllib.error.URLError) as exc:
                if attempt == retries:
                    errors.append(exc)
                    return
                time.sleep(min(30, 2 + attempt))

    threads = [threading.Thread(target=worker, args=(s,), daemon=True) for s in segments]
    for t in threads:
        t.start()
    next_report = 0.0
    while any(t.is_alive() for t in threads):
        time.sleep(5)
        with lock:
            done = sum(s["pos"] - s["start"] for s in segments) + segments[0]["start"]
        if done / size >= next_report:
            print(f"  {dest.name}: {100 * done / size:.0f}% of {size / 2**20:.0f} MiB ({connections} connections)", flush=True)
            next_report = done / size + 0.05
    if errors or any(s["pos"] < s["end"] for s in segments):
        raise IOError(f"{dest.name}: incomplete after retries ({errors[:1]})")
    if md5 is not None and file_md5(tmp) != md5:
        raise IOError(f"{dest.name}: MD5 mismatch")
    tmp.replace(dest)
    state_path.unlink(missing_ok=True)
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
