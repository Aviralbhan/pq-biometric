"""Live demo helpers: one command per step of docs/DEMO.md.

    python scripts/demo.py server              # terminal 1: verification server (in-process Redis), port 8000
    python scripts/demo.py health              # every module and whether it is real or a stub
    python scripts/demo.py enrol               # webcam reference photo (SPACE) + device keys for PQ and CLASSICAL
    python scripts/demo.py verify PQ           # genuine transaction; also CLASSICAL. Prints the signed verdict
    python scripts/demo.py replay              # re-submit / flip a byte / swap the transaction -> REPLAY, TAMPER, TX_MISMATCH
    python scripts/demo.py morph               # M2 on one FRLL-Morphs pair and one genuine FRLL pair
    python scripts/demo.py deepfake            # M3 on 3 real and 3 fake OpenForensics faces
    python scripts/demo.py injection           # verify through OBS Virtual Camera (skipped if OBS is not present)
    python scripts/demo.py verdict             # offline signature check passes; edit a score in a copy -> fails
    python scripts/demo.py crypto              # ML-KEM/ML-DSA vs X25519/Ed25519 times and bytes (eval/results)

Demo state lives in var/demo/ (config, server keys, database, device keys, last session).
Set PQBIO_OFFLINE_GUARD=1 to make every network connection except localhost fail (reliability check).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yaml  # noqa: E402

from settings import load_config, resolve  # noqa: E402

DEMO = ROOT / "var" / "demo"
CONFIG = DEMO / "config.yaml"
LAST = DEMO / "last"
USER = "aviral"
TX = {"type": "payment", "amount": 500, "currency": "INR", "payee": "acct-42"}
TX_OTHER = {"type": "payment", "amount": 50000, "currency": "INR", "payee": "attacker"}
PORT = 8000


def demo_config() -> dict:
    """config.yaml with demo paths: everything under var/demo, server on 127.0.0.1:8000."""
    cfg = load_config(overrides={
        "server": {"host": "127.0.0.1", "port": PORT, "keys_dir": "var/demo/keys", "db_path": "var/demo/server.sqlite",
                   "log_path": "var/demo/server.log"},
        "redis": {"url": "fakeredis://"},
        "capture": {"server_url": f"http://127.0.0.1:{PORT}", "device_dir": "var/demo/device"},
    })
    DEMO.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return cfg


def run(*args: str) -> int:
    return subprocess.call([sys.executable, *args], cwd=ROOT)


def db() -> sqlite3.Connection:
    return sqlite3.connect(f"file:{resolve(demo_config()['server']['db_path'])}?mode=ro", uri=True)


def show_verdict(challenge_id: str | None = None) -> dict | None:
    """Print the signed verdict (latest, or for one challenge), its scores against the thresholds, and the signature check."""
    from crypto import Mode
    from server.verdict import signer_id, verify_record

    cfg = demo_config()
    with db() as conn:
        row = (conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts WHERE challenge_id = ?", (challenge_id,)).fetchone()
               if challenge_id else conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts ORDER BY created_ms DESC LIMIT 1").fetchone())
    if not row:
        print("no verdict stored yet")
        return None
    cid, mode, record, signature = row
    rec = json.loads(record)
    pub = (resolve(cfg["server"]["keys_dir"]) / f"verdict_{mode}.pub").read_bytes()
    ok = verify_record(Mode(mode), pub, record, signature) and rec["signer"] == signer_id(pub)
    t = cfg["thresholds"]
    print(f"\n  verdict {cid} [{mode}]: {rec['decision'].upper()}  reason: {rec['reason_code'] or '-'}")
    print(f"  {'module':13} {'score':>8}  {'threshold':>9}  rule")
    for key, label in (("match", "matcher"), ("m1_pad", "M1 PAD"), ("m2_morph", "M2 morph"), ("m3_deepfake", "M3 deepfake"),
                       ("m4_replay", "M4 actions"), ("m5_injection", "M5 injection")):
        score = rec["scores"].get(key)
        rule = "accept if >=" if key == "match" else "reject if >"
        print(f"  {label:13} {'-' if score is None else f'{score:8.3f}':>8}  {t[key]:>9.4f}  {rule}")
    print(f"  thresholds {rec['thresholds_version']}; ML-DSA/Ed25519 signature over the record: {'VALID' if ok else 'INVALID'}")
    return rec


def cmd_server(args) -> int:
    demo_config()
    print(f"config -> {CONFIG}; starting server on http://127.0.0.1:{PORT} (Ctrl+C to stop)")
    return run("-m", "server", "--fakeredis", "--config", str(CONFIG), "--port", str(PORT))


def cmd_health(args) -> int:
    import httpx

    h = httpx.get(f"http://127.0.0.1:{PORT}/health", timeout=10).json()
    print(f"status {h['status']}, modes {h['allowed_modes']}, thresholds {h['thresholds_version']}")
    for module, version in h["model_versions"].items():
        print(f"  {module:14} {version:32} {'STUB' if version == 'stub' else 'real'}")
    print("all modules real" if not h["stubs"] else f"STUBS: {h['stubs']}")
    return 0 if not h["stubs"] else 1


def cmd_enrol(args) -> int:
    demo_config()
    return run("-m", "client.enroll", "--user", USER, "--config", str(CONFIG))


def cmd_verify(args) -> int:
    """Capture and verify. An accepted attempt becomes var/demo/last, the genuine bundle the replay step re-uses."""
    demo_config()
    attempt = DEMO / "attempt"
    if attempt.exists():
        shutil.rmtree(attempt)
    code = run("-m", "client.capture", "--user", USER, "--mode", args.mode, "--config", str(CONFIG), "--save", str(attempt),
               "--tx", json.dumps(TX), *(["--camera", str(args.camera)] if args.camera is not None else []))
    ch = json.loads((attempt / "challenge.json").read_text(encoding="utf-8"))["challenge"] if (attempt / "challenge.json").exists() else {}
    rec = show_verdict(ch.get("challenge_id"))
    if rec and rec["decision"] == "accepted":
        if LAST.exists():
            shutil.rmtree(LAST)
        attempt.rename(LAST)
    return code


def cmd_replay(args) -> int:
    from client.capture import load_session
    from client.protocol import ApiClient, DeviceIdentity, check_challenge, make_manifest, seal_evidence

    cfg = demo_config()
    if not (LAST / "bundle.bin").exists():
        print("run 'python scripts/demo.py verify PQ' first (it saves the genuine bundle)")
        return 1
    saved = json.loads((LAST / "challenge.json").read_text(encoding="utf-8"))
    api = ApiClient(cfg["capture"]["server_url"])
    identity = DeviceIdentity.load(resolve(cfg["capture"]["device_dir"]), USER, saved["mode"])
    video, meta = load_session(LAST)

    def reason(request_id: int) -> str:
        with db() as conn:
            return conn.execute("SELECT reason_code FROM requests WHERE request_id = ?", (request_id,)).fetchone()[0] or "accepted"

    print("1. Re-submit the captured genuine bundle unchanged (an eavesdropper's replay):")
    resp, rid = api.verify_bytes((LAST / "bundle.bin").read_bytes())
    print(f"   client sees: {resp['decision']}; server reason: {reason(rid)}")

    print("2. Fresh challenge, genuine evidence, one bit flipped in the encrypted payload (tampering in transit):")
    ch = api.challenge(USER, TX, identity.mode)
    check_challenge(identity, ch, TX)
    bundle = seal_evidence(identity, ch, video, meta)
    payload = bytearray(bundle.payload)
    payload[len(payload) // 2] ^= 0x01
    resp, rid = api.verify(replace(bundle, payload=bytes(payload)))
    print(f"   client sees: {resp['decision']}; server reason: {reason(rid)}")

    print("3. The same evidence re-bound to a different transaction (attacker with the device key moves 50,000 INR):")
    ch_other = api.challenge(USER, TX_OTHER, identity.mode)
    manifest = replace(make_manifest(identity, ch_other, video), tx_hash=saved["challenge"]["tx_hash"])
    resp, rid = api.verify(seal_evidence(identity, ch_other, video, meta, manifest=manifest))
    print(f"   client sees: {resp['decision']}; server reason: {reason(rid)}")
    return 0


def cmd_morph(args) -> int:
    from detectors.m2_morph.dmad import DMADMorph
    from scripts.train_m2 import Embedder

    cfg = demo_config()
    embed = Embedder(cfg)
    m2 = DMADMorph(resolve(cfg["detectors"]["m2_morph"]["model"]))
    thr = cfg["thresholds"]["m2_morph"]
    frll = ROOT / "data" / "frll"
    pairs = [
        ("genuine: subject 001 neutral photo enrolled, 001 smiling probe",
         frll / "neutral_front" / "neutral_front" / "001_03.jpg", frll / "smiling_front" / "smiling_front" / "001_08.jpg"),
        ("attack: morph of 001 + 002 enrolled (AMSL), 001 smiling probe",
         ROOT / "data" / "frll_morphs" / "facelab_london" / "morph_amsl" / "001_002.jpg", frll / "smiling_front" / "smiling_front" / "001_08.jpg"),
    ]
    print(f"M2 {m2.version}; reject if score > {thr:.4f} (BPCER 0.5%)")
    for label, ref, probe in pairs:
        r, p = embed(str(ref)), embed(str(probe))
        if r is None or p is None:
            print(f"  {label}: no face found")
            continue
        score = m2.score_pair(r, p)
        print(f"  {label}\n     score {score:.4f} -> {'REJECT (MORPH_FLAG)' if score > thr else 'accept'}")
    return 0


def cmd_deepfake(args) -> int:
    import cv2

    from detectors.m3_deepfake.gend import GenDDeepfake

    cfg = demo_config()
    c = cfg["detectors"]["m3_deepfake"]
    m3 = GenDDeepfake(resolve(c["model"]), c["version"], c.get("fake_index", 1), c.get("threads", 8))
    thr = cfg["thresholds"]["m3_deepfake"]
    split_csv = next((p for p in (ROOT / "data" / "splits" / "m3_test-dev.csv", ROOT / "data" / "splits" / "m3_val.csv") if p.exists()), None)
    if split_csv is None:
        print("no OpenForensics crops; run scripts/download_openforensics.py")
        return 1
    with open(split_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(7)
    picks = rng.sample([r for r in rows if r["label"] == "bona_fide"], 3) + rng.sample([r for r in rows if r["label"] == "attack"], 3)
    probs = m3.p_fake([cv2.resize(cv2.imread(r["path"]), (224, 224)) for r in picks])
    print(f"M3 {m3.version} on {split_csv.name}; reject if P(fake) > {thr:.4f}")
    for r, p in zip(picks, probs):
        truth = "real" if r["label"] == "bona_fide" else "FAKE"
        print(f"  {truth:4}  P(fake) {p:.4f} -> {'REJECT (DEEPFAKE_FAIL)' if p > thr else 'accept'}   {Path(r['path']).name}")
    return 0


def cmd_injection(args) -> int:
    from client.capture import device_name

    names = []
    for i in range(6):
        n = device_name(i)
        if n:
            names.append((i, n))
    obs = [(i, n) for i, n in names if "obs" in n.lower()]
    print("cameras: " + ", ".join(f"{i}: {n}" for i, n in names))
    if not obs:
        print("OBS Virtual Camera not present: step skipped (install OBS and start its virtual camera to show INJECTION_FAIL)")
        return 0
    args.mode, args.camera = "PQ", obs[0][0]
    print(f"verifying through camera {obs[0][0]} ({obs[0][1]}) - expect INJECTION_FAIL")
    return cmd_verify(args)


def cmd_verdict(args) -> int:
    from crypto import Mode, canonical_json
    from server.verdict import verify_record

    cfg = demo_config()
    with db() as conn:  # the latest accepted transaction: a verdict that carries every detector score
        row = conn.execute("SELECT challenge_id FROM verdicts WHERE json_extract(record, '$.decision') = 'accepted' "
                           "ORDER BY created_ms DESC LIMIT 1").fetchone()
    if row is None:
        print("no accepted verdict yet: run 'python scripts/demo.py verify PQ' first")
        return 1
    target = row[0]
    rec = show_verdict(target)
    src = resolve(cfg["server"]["db_path"])
    copy = DEMO / "tampered_copy.sqlite"
    with sqlite3.connect(f"file:{src}?mode=ro", uri=True) as conn, sqlite3.connect(copy) as out:
        conn.backup(out)
    with sqlite3.connect(copy) as conn:
        cid, mode, record, signature = conn.execute(
            "SELECT challenge_id, mode, record, signature FROM verdicts WHERE challenge_id = ?", (target,)).fetchone()
        edited = json.loads(record)
        old = edited["scores"]["m3_deepfake"]
        edited["scores"]["m3_deepfake"] = 0.01 if old and old > 0.5 else 0.99  # e.g. hide or invent a deepfake flag
        conn.execute("UPDATE verdicts SET record = ? WHERE challenge_id = ?", (canonical_json(edited), cid))
        _, record2, signature2 = conn.execute("SELECT mode, record, signature FROM verdicts WHERE challenge_id = ?", (cid,)).fetchone()
    pub = (resolve(cfg["server"]["keys_dir"]) / f"verdict_{mode}.pub").read_bytes()
    print(f"\n  copy of the database: M3 score in that verdict edited {old} -> {edited['scores']['m3_deepfake']}")
    print(f"  offline check of the edited record: {'VALID' if verify_record(Mode(mode), pub, record2, signature2) else 'INVALID - edit detected'}")
    print(f"  (original database untouched; the copy is {copy.relative_to(ROOT)})")
    return 0


def cmd_crypto(args) -> int:
    path, sizes = ROOT / "eval" / "results" / "crypto_bench.csv", ROOT / "eval" / "results" / "crypto_sizes.csv"
    if not path.exists() or not sizes.exists():
        print("eval/results/crypto_bench.csv missing: run 'python -m eval.crypto_bench --runs 1000' on an idle machine")
        return 1
    with open(path, encoding="utf-8") as fh:
        rows = {(r["mode"], r["op"]): r for r in csv.DictReader(fh)}
    with open(sizes, encoding="utf-8") as fh:
        size = {r["mode"]: r for r in csv.DictReader(fh)}
    any_row = next(iter(rows.values()))
    print(f"median of {any_row['runs']} runs per operation; {any_row['system']}, cryptography {any_row['cryptography']}, payload {any_row['payload_kb']} KB")
    print(f"  {'operation':18} {'X25519/Ed25519':>15} {'ML-KEM/ML-DSA':>15}")
    for op, label in (("encapsulate", "key exchange (client)"), ("decapsulate", "key exchange (server)"), ("sign_manifest", "sign manifest"),
                      ("verify_manifest", "verify manifest"), ("sign_verdict", "sign verdict"), ("seal_bundle", "seal bundle"), ("unseal_bundle", "unseal bundle")):
        print(f"  {label:22} {float(rows[('CLASSICAL', op)]['p50_us']):>9.1f} us {float(rows[('PQ', op)]['p50_us']):>11.1f} us")
    pq, cl = size["PQ"], size["CLASSICAL"]
    extra = int(pq["per_transaction_asymmetric_B"]) - int(cl["per_transaction_asymmetric_B"])
    print(f"  bytes per request: KEM ciphertext {cl['kem_ciphertext_B']} -> {pq['kem_ciphertext_B']}, "
          f"signature {cl['signature_B']} -> {pq['signature_B']}; PQ adds {extra:,} bytes")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("server", "health", "enrol", "replay", "morph", "deepfake", "verdict", "crypto"):
        sub.add_parser(name)
    v = sub.add_parser("verify")
    v.add_argument("mode", choices=["PQ", "CLASSICAL"])
    v.add_argument("--camera", type=int)
    inj = sub.add_parser("injection")
    inj.set_defaults(camera=None)
    args = ap.parse_args(argv)
    os.chdir(ROOT)
    return globals()[f"cmd_{args.cmd}"](args)


if __name__ == "__main__":
    raise SystemExit(main())
