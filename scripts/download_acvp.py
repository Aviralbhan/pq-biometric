"""Download NIST ACVP known-answer vectors for ML-KEM-768 and ML-DSA-65.

Source: github.com/usnistgov/ACVP-Server (gen-val/json-files). Only the test
groups for our parameter sets are kept, written to tests/vectors/ as small JSON
files so the crypto tests can run offline.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

BASE = "https://raw.githubusercontent.com/usnistgov/ACVP-Server/master/gen-val/json-files"
OUT_DIR = Path(__file__).resolve().parent.parent / "tests" / "vectors"


def _sigver_usable(group: dict) -> bool:
    # `cryptography` exposes pure ML-DSA with context (external) and verify_mu
    # (external mu); HashML-DSA and internal-without-mu have no public API.
    if group.get("signatureInterface") == "external":
        return group.get("preHash") == "pure"
    return bool(group.get("externalMu"))


# (ACVP folder, parameter set, output file, extra group filter)
SOURCES = [
    ("ML-KEM-keyGen-FIPS203", "ML-KEM-768", "mlkem768_keygen.json", None),
    ("ML-DSA-keyGen-FIPS204", "ML-DSA-65", "mldsa65_keygen.json", None),
    ("ML-DSA-sigVer-FIPS204", "ML-DSA-65", "mldsa65_sigver.json", _sigver_usable),
]


def fetch(folder: str) -> dict:
    url = f"{BASE}/{folder}/internalProjection.json"
    with urllib.request.urlopen(url, timeout=60) as resp:
        return json.load(resp)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for folder, param_set, out_name, keep in SOURCES:
        data = fetch(folder)
        groups = [
            g for g in data["testGroups"]
            if g.get("parameterSet") == param_set and (keep is None or keep(g))
        ]
        if not groups:
            print(f"no {param_set} groups in {folder}", file=sys.stderr)
            return 1
        out = {"source": f"usnistgov/ACVP-Server {folder}", "vsId": data.get("vsId"), "testGroups": groups}
        (OUT_DIR / out_name).write_text(json.dumps(out, indent=1), encoding="utf-8")
        n = sum(len(g["tests"]) for g in groups)
        print(f"{out_name}: {len(groups)} groups, {n} tests")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
