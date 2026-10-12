"""Load config.yaml. Every path in the config is resolved against the repo root."""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "config.yaml"


def install_offline_guard() -> None:
    """Refuse every network connection except to this machine (PQBIO_OFFLINE_GUARD=1).

    Used by the demo reliability check to prove nothing is downloaded at runtime.
    Hugging Face libraries are also told to stay offline.
    """
    import ipaddress
    import socket

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    original = socket.socket.connect

    def guarded_connect(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        try:
            local = host == "localhost" or ipaddress.ip_address(host).is_loopback
        except ValueError:
            local = False
        if not local:
            raise ConnectionRefusedError(f"PQBIO_OFFLINE_GUARD: blocked network connection to {host!r}")
        return original(sock, address)

    socket.socket.connect = guarded_connect


if os.environ.get("PQBIO_OFFLINE_GUARD") == "1":
    install_offline_guard()


def load_config(path: str | os.PathLike[str] | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Read the YAML config (PQBIO_CONFIG env var, else config.yaml) and deep-merge overrides.

    PQBIO_REDIS_URL, if set, replaces redis.url (docker-compose points it at the redis service).
    """
    path = Path(path or os.environ.get("PQBIO_CONFIG") or DEFAULT_CONFIG)
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if os.environ.get("PQBIO_REDIS_URL"):
        cfg["redis"]["url"] = os.environ["PQBIO_REDIS_URL"]
    if overrides:
        cfg = deep_merge(cfg, overrides)
    return cfg


def deep_merge(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def resolve(path: str | os.PathLike[str]) -> Path:
    """Absolute paths are kept; relative ones are taken from the repo root."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p
