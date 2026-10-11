"""Video container shared by client and server: a sequence of JPEG frames.

Format: b"BEAV1" || uint32 frame_count || (uint32 length || JPEG bytes) * frame_count.
JPEG frames decode from memory with cv2.imdecode (no temp files), and the bytes
are exactly what the client hashed into sample_hash.
"""

from __future__ import annotations

import struct
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

MAGIC = b"BEAV1"
_U32 = struct.Struct(">I")


def encode_jpeg(frame: np.ndarray, quality: int = 90) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()


def pack_jpegs(jpegs: Sequence[bytes]) -> bytes:
    parts = [MAGIC, _U32.pack(len(jpegs))]
    for jpg in jpegs:
        parts += [_U32.pack(len(jpg)), jpg]
    return b"".join(parts)


def encode_frames(frames: Iterable[np.ndarray], quality: int = 90) -> bytes:
    return pack_jpegs([encode_jpeg(f, quality) for f in frames])


def unpack_jpegs(video: bytes) -> list[bytes]:
    if not video.startswith(MAGIC):
        raise ValueError("not a BEAV1 video")
    offset = len(MAGIC)
    (count,) = _U32.unpack_from(video, offset)
    offset += _U32.size
    out: list[bytes] = []
    for _ in range(count):
        if offset + _U32.size > len(video):
            raise ValueError("truncated video")
        (n,) = _U32.unpack_from(video, offset)
        offset += _U32.size
        if offset + n > len(video):
            raise ValueError("truncated video")
        out.append(video[offset : offset + n])
        offset += n
    if offset != len(video):
        raise ValueError("trailing bytes in video")
    return out


def decode_jpeg(jpg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("JPEG decoding failed")
    return img


def decode_frames(video: bytes, indices: Sequence[int] | None = None) -> list[np.ndarray]:
    jpegs = unpack_jpegs(video)
    if indices is not None:
        jpegs = [jpegs[i] for i in indices]
    return [decode_jpeg(j) for j in jpegs]


def sample_indices(n: int, k: int) -> list[int]:
    """k indices spread evenly over range(n) (all of them if n <= k)."""
    if n <= 0 or k <= 0:
        return []
    if n <= k:
        return list(range(n))
    return [int(round(i * (n - 1) / (k - 1))) for i in range(k)] if k > 1 else [n // 2]
