"""Run the verification server.

    python -m server                       # config.yaml, Redis at redis.url
    python -m server --fakeredis           # in-process Redis (Windows dev without a Redis install)
    python -m server --config other.yaml --port 8001
"""

from __future__ import annotations

import argparse
import os

import uvicorn

from settings import load_config


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--fakeredis", action="store_true")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    args = ap.parse_args(argv)
    if args.config:
        os.environ["PQBIO_CONFIG"] = args.config
    if args.fakeredis:
        os.environ["PQBIO_REDIS_URL"] = "fakeredis://"
    cfg = load_config()
    uvicorn.run("server.app:app", host=args.host or cfg["server"]["host"], port=args.port or cfg["server"]["port"], workers=1)


if __name__ == "__main__":
    main()
