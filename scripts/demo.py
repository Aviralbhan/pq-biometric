"""Live demo helpers: one command per step of docs/DEMO.md (the demo web page at /ui calls the same functions).

    python scripts/demo.py server              # terminal 1: verification server (in-process Redis), port 8000, demo page /ui
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
Each step has a *_data() function returning plain data (used by the web page) and a cmd_* printer (CLI).
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
MODULES = (("match", "matcher"), ("m1_pad", "M1 PAD"), ("m2_morph", "M2 morph"), ("m3_deepfake", "M3 deepfake"),
           ("m4_replay", "M4 actions"), ("m5_injection", "M5 injection"))
CRYPTO_OPS = (("encapsulate", "key exchange (client)"), ("decapsulate", "key exchange (server)"), ("sign_manifest", "sign manifest"),
              ("verify_manifest", "verify manifest"), ("sign_verdict", "sign verdict"), ("seal_bundle", "seal bundle"),
              ("unseal_bundle", "unseal bundle"))


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


# ---- data functions (shared by the CLI and the web page) ----------------------------------


def verdict_data(challenge_id: str | None = None) -> dict | None:
    """The signed verdict (one challenge, or the latest), the thresholds, and whether its signature verifies."""
    from crypto import Mode
    from server.verdict import signer_id, verify_record

    cfg = demo_config()
    with db() as conn:
        row = (conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts WHERE challenge_id = ?", (challenge_id,)).fetchone()
               if challenge_id else conn.execute("SELECT challenge_id, mode, record, signature FROM verdicts ORDER BY created_ms DESC LIMIT 1").fetchone())
    if not row:
        return None
    cid, mode, record, signature = row
    rec = json.loads(record)
    pub = (resolve(cfg["server"]["keys_dir"]) / f"verdict_{mode}.pub").read_bytes()
    ok = verify_record(Mode(mode), pub, record, signature) and rec["signer"] == signer_id(pub)
    return {"challenge_id": cid, "mode": mode, "record": rec, "signature_valid": ok, "thresholds": cfg["thresholds"]}


def request_data(request_id: int) -> dict | None:
    """One row of the server's request log: decision, reason code, bytes received, per-stage timing (microseconds)."""
    with db() as conn:
        row = conn.execute("SELECT request_id, challenge_id, mode, decision, reason_code, bytes_in, timing_json FROM requests "
                           "WHERE request_id = ?", (request_id,)).fetchone()
    if not row:
        return None
    keys = ("request_id", "challenge_id", "mode", "decision", "reason_code", "bytes_in", "timing")
    out = dict(zip(keys, row))
    out["timing"] = json.loads(out["timing"])
    return out


def latest_request_id() -> int:
    with db() as conn:
        row = conn.execute("SELECT MAX(request_id) FROM requests").fetchone()
    return int(row[0] or 0)


def verify_capture(mode: str, camera: int | None = None) -> tuple[int, str | None]:
    """Capture and verify with the client. An accepted attempt becomes var/demo/last (the replay step re-uses it).

    Returns (client exit code, challenge_id or None)."""
    demo_config()
    attempt = DEMO / "attempt"
    if attempt.exists():
        shutil.rmtree(attempt)
    code = run("-m", "client.capture", "--user", USER, "--mode", mode, "--config", str(CONFIG), "--save", str(attempt),
               "--tx", json.dumps(TX), *(["--camera", str(camera)] if camera is not None else []))
    ch = json.loads((attempt / "challenge.json").read_text(encoding="utf-8"))["challenge"] if (attempt / "challenge.json").exists() else {}
    data = verdict_data(ch.get("challenge_id")) if ch else None
    if data and data["record"]["decision"] == "accepted":
        if LAST.exists():
            shutil.rmtree(LAST)
        attempt.rename(LAST)
    return code, ch.get("challenge_id")


def replay_data() -> list[dict]:
    """The three protocol attacks on the last genuine bundle. One dict per attack with the server's request id."""
    from client.capture import load_session
    from client.protocol import ApiClient, DeviceIdentity, check_challenge, make_manifest, seal_evidence

    cfg = demo_config()
    if not (LAST / "bundle.bin").exists():
        raise FileNotFoundError("no genuine bundle yet: run 'verify' first (it saves the bundle this step re-uses)")
    saved = json.loads((LAST / "challenge.json").read_text(encoding="utf-8"))
    api = ApiClient(cfg["capture"]["server_url"])
    identity = DeviceIdentity.load(resolve(cfg["capture"]["device_dir"]), USER, saved["mode"])
    video, meta = load_session(LAST)
    out = []

    original = (LAST / "bundle.bin").read_bytes()
    resp, rid = api.verify_bytes(original)
    out.append({"attack": "Re-submit the captured genuine bundle unchanged (an eavesdropper's replay)", "client_sees": resp["decision"], "request_id": rid,
                "changed": f"nothing: the same {len(original):,} bytes, sent a second time (challenge {saved['challenge']['challenge_id'][:8]}...)"})

    ch = api.challenge(USER, TX, identity.mode)
    check_challenge(identity, ch, TX)
    bundle = seal_evidence(identity, ch, video, meta)
    payload = bytearray(bundle.payload)
    flip_at = len(payload) // 2
    before = payload[flip_at]
    payload[flip_at] ^= 0x01
    resp, rid = api.verify(replace(bundle, payload=bytes(payload)))
    out.append({"attack": "Fresh challenge, genuine evidence, one bit flipped in the encrypted payload (tampering in transit)",
                "client_sees": resp["decision"], "request_id": rid,
                "changed": f"1 bit at byte {flip_at:,} of the {len(payload):,}-byte encrypted payload (0x{before:02x} -> 0x{payload[flip_at]:02x})"})

    ch_other = api.challenge(USER, TX_OTHER, identity.mode)
    manifest = replace(make_manifest(identity, ch_other, video), tx_hash=saved["challenge"]["tx_hash"])
    resp, rid = api.verify(seal_evidence(identity, ch_other, video, meta, manifest=manifest))
    out.append({"attack": "The same evidence re-bound to a different transaction (attacker with the device key moves 50,000 INR)",
                "client_sees": resp["decision"], "request_id": rid,
                "changed": f"transaction: {TX['amount']:,} {TX['currency']} to {TX['payee']} -> {TX_OTHER['amount']:,} {TX_OTHER['currency']} to "
                           f"{TX_OTHER['payee']}; the signed evidence still names the original one"})
    for row in out:
        row["server_reason"] = (request_data(row["request_id"]) or {}).get("reason_code") or "accepted"
    return out


def morph_data(embed=None) -> dict:
    from detectors.m2_morph.dmad import DMADMorph
    from scripts.train_m2 import Embedder

    cfg = demo_config()
    embed = embed or Embedder(cfg)
    m2 = DMADMorph(resolve(cfg["detectors"]["m2_morph"]["model"]))
    thr = cfg["thresholds"]["m2_morph"]
    frll = ROOT / "data" / "frll"
    pairs = [
        ("genuine: subject 001 neutral photo enrolled, 001 smiling probe",
         frll / "neutral_front" / "neutral_front" / "001_03.jpg", frll / "smiling_front" / "smiling_front" / "001_08.jpg"),
        ("attack: morph of 001 + 002 enrolled (AMSL), 001 smiling probe",
         ROOT / "data" / "frll_morphs" / "facelab_london" / "morph_amsl" / "001_002.jpg", frll / "smiling_front" / "smiling_front" / "001_08.jpg"),
    ]
    rows = []
    for label, ref, probe in pairs:
        r, p = embed(str(ref)), embed(str(probe))
        score = None if r is None or p is None else m2.score_pair(r, p)
        rows.append({"pair": label, "score": score, "rejected": score is not None and score > thr,
                     "reference": str(Path(ref).relative_to(ROOT).as_posix()), "probe": str(Path(probe).relative_to(ROOT).as_posix())})
    return {"model": m2.version, "threshold": thr, "pairs": rows}


def deepfake_data(m3=None) -> dict:
    import cv2

    from detectors.m3_deepfake.gend import GenDDeepfake

    cfg = demo_config()
    c = cfg["detectors"]["m3_deepfake"]
    m3 = m3 or GenDDeepfake(resolve(c["model"]), c["version"], c.get("fake_index", 1), c.get("threads", 8))
    thr = cfg["thresholds"]["m3_deepfake"]
    split_csv = next((p for p in (ROOT / "data" / "splits" / "m3_test-dev.csv", ROOT / "data" / "splits" / "m3_val.csv") if p.exists()), None)
    if split_csv is None:
        raise FileNotFoundError("no OpenForensics crops; run scripts/download_openforensics.py")
    with open(split_csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    rng = random.Random(7)
    picks = rng.sample([r for r in rows if r["label"] == "bona_fide"], 3) + rng.sample([r for r in rows if r["label"] == "attack"], 3)
    probs = m3.p_fake([cv2.resize(cv2.imread(r["path"]), (224, 224)) for r in picks])
    faces = [{"truth": "real" if r["label"] == "bona_fide" else "fake", "p_fake": float(p), "rejected": bool(p > thr),
              "file": Path(r["path"]).name, "path": Path(r["path"]).resolve().relative_to(ROOT).as_posix()} for r, p in zip(picks, probs)]
    return {"model": m3.version, "dataset": split_csv.name, "threshold": thr, "faces": faces}


def cameras() -> list[tuple[int, str]]:
    from client.capture import device_name

    found = []
    for i in range(6):
        n = device_name(i)
        if n:
            found.append((i, n))
    return found


def obs_camera() -> tuple[int, str] | None:
    return next(((i, n) for i, n in cameras() if "obs" in n.lower()), None)


def verdict_tamper_data() -> dict:
    """Verify the latest accepted verdict offline, then edit one score in a copy of the database and verify again."""
    from crypto import Mode, canonical_json
    from server.verdict import verify_record

    cfg = demo_config()
    with db() as conn:  # the latest accepted transaction: a verdict that carries every detector score
        row = conn.execute("SELECT challenge_id FROM verdicts WHERE json_extract(record, '$.decision') = 'accepted' "
                           "ORDER BY created_ms DESC LIMIT 1").fetchone()
    if row is None:
        raise LookupError("no accepted verdict yet: run 'verify' first")
    target = row[0]
    original = verdict_data(target)
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
    return {"original": original, "edited_field": "m3_deepfake", "old_value": old, "new_value": edited["scores"]["m3_deepfake"],
            "edited_valid": verify_record(Mode(mode), pub, record2, signature2), "copy": str(copy.relative_to(ROOT))}


def crypto_data() -> dict | None:
    path, sizes = ROOT / "eval" / "results" / "crypto_bench.csv", ROOT / "eval" / "results" / "crypto_sizes.csv"
    if not path.exists() or not sizes.exists():
        return None
    with open(path, encoding="utf-8") as fh:
        rows = {(r["mode"], r["op"]): r for r in csv.DictReader(fh)}
    with open(sizes, encoding="utf-8") as fh:
        size = {r["mode"]: r for r in csv.DictReader(fh)}
    any_row = next(iter(rows.values()))
    pq, cl = size["PQ"], size["CLASSICAL"]
    return {
        "runs": any_row["runs"], "system": any_row["system"], "cryptography": any_row["cryptography"], "payload_kb": any_row["payload_kb"],
        "ops": [{"op": label, "classical_us": float(rows[("CLASSICAL", op)]["p50_us"]), "pq_us": float(rows[("PQ", op)]["p50_us"])}
                for op, label in CRYPTO_OPS],
        "sizes": {"kem_ciphertext": [cl["kem_ciphertext_B"], pq["kem_ciphertext_B"]], "signature": [cl["signature_B"], pq["signature_B"]],
                  "extra_per_request": int(pq["per_transaction_asymmetric_B"]) - int(cl["per_transaction_asymmetric_B"])},
    }


# ---- CLI ----------------------------------------------------------------------------------


def show_verdict(challenge_id: str | None = None) -> dict | None:
    """Print the signed verdict (latest, or for one challenge), its scores against the thresholds, and the signature check."""
    data = verdict_data(challenge_id)
    if data is None:
        print("no verdict stored yet")
        return None
    rec, t = data["record"], data["thresholds"]
    print(f"\n  verdict {data['challenge_id']} [{data['mode']}]: {rec['decision'].upper()}  reason: {rec['reason_code'] or '-'}")
    print(f"  {'module':13} {'score':>8}  {'threshold':>9}  rule")
    for key, label in MODULES:
        score = rec["scores"].get(key)
        rule = "accept if >=" if key == "match" else "reject if >"
        print(f"  {label:13} {'-' if score is None else f'{score:8.3f}':>8}  {t[key]:>9.4f}  {rule}")
    print(f"  thresholds {rec['thresholds_version']}; ML-DSA/Ed25519 signature over the record: {'VALID' if data['signature_valid'] else 'INVALID'}")
    return rec


def cmd_server(args) -> int:
    demo_config()
    os.environ["PQBIO_DEMO_UI"] = "1"  # demo endpoints and the /ui page exist only in demo mode
    print(f"config -> {CONFIG}; starting server on http://127.0.0.1:{PORT}, demo page http://127.0.0.1:{PORT}/ui (Ctrl+C to stop)")
    return run("-m", "server", "--fakeredis", "--config", str(CONFIG), "--port", str(PORT), "--host", "127.0.0.1")


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
    code, challenge_id = verify_capture(args.mode, args.camera)
    show_verdict(challenge_id)
    return code


def cmd_replay(args) -> int:
    try:
        rows = replay_data()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    for i, row in enumerate(rows, 1):
        print(f"{i}. {row['attack']}:")
        print(f"   client sees: {row['client_sees']}; server reason: {row['server_reason']}")
    return 0


def cmd_morph(args) -> int:
    d = morph_data()
    print(f"M2 {d['model']}; reject if score > {d['threshold']:.4f} (BPCER 0.5%)")
    for p in d["pairs"]:
        if p["score"] is None:
            print(f"  {p['pair']}: no face found")
        else:
            print(f"  {p['pair']}\n     score {p['score']:.4f} -> {'REJECT (MORPH_FLAG)' if p['rejected'] else 'accept'}")
    return 0


def cmd_deepfake(args) -> int:
    try:
        d = deepfake_data()
    except FileNotFoundError as exc:
        print(exc)
        return 1
    print(f"M3 {d['model']} on {d['dataset']}; reject if P(fake) > {d['threshold']:.4f}")
    for f in d["faces"]:
        print(f"  {f['truth'].upper() if f['truth'] == 'fake' else f['truth']:4}  P(fake) {f['p_fake']:.4f} -> "
              f"{'REJECT (DEEPFAKE_FAIL)' if f['rejected'] else 'accept'}   {f['file']}")
    return 0


def cmd_injection(args) -> int:
    print("cameras: " + ", ".join(f"{i}: {n}" for i, n in cameras()))
    obs = obs_camera()
    if not obs:
        print("OBS Virtual Camera not present: step skipped (install OBS and start its virtual camera to show INJECTION_FAIL)")
        return 0
    print(f"verifying through camera {obs[0]} ({obs[1]}) - expect INJECTION_FAIL")
    code, challenge_id = verify_capture("PQ", obs[0])
    show_verdict(challenge_id)
    return code


def cmd_verdict(args) -> int:
    try:
        d = verdict_tamper_data()
    except LookupError as exc:
        print(exc)
        return 1
    show_verdict(d["original"]["challenge_id"])
    print(f"\n  copy of the database: M3 deepfake score in that verdict edited {d['old_value']} -> {d['new_value']}")
    print(f"  offline check of the edited record: {'VALID' if d['edited_valid'] else 'INVALID - edit detected'}")
    print(f"  (original database untouched; the copy is {d['copy']})")
    return 0


def cmd_crypto(args) -> int:
    d = crypto_data()
    if d is None:
        print("eval/results/crypto_bench.csv missing: run 'python -m eval.crypto_bench --runs 1000' on an idle machine")
        return 1
    print(f"median of {d['runs']} runs per operation; {d['system']}, cryptography {d['cryptography']}, payload {d['payload_kb']} KB")
    print(f"  {'operation':18} {'X25519/Ed25519':>15} {'ML-KEM/ML-DSA':>15}")
    for o in d["ops"]:
        print(f"  {o['op']:22} {o['classical_us']:>9.1f} us {o['pq_us']:>11.1f} us")
    s = d["sizes"]
    print(f"  bytes per request: KEM ciphertext {s['kem_ciphertext'][0]} -> {s['kem_ciphertext'][1]}, "
          f"signature {s['signature'][0]} -> {s['signature'][1]}; PQ adds {s['extra_per_request']:,} bytes")
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
