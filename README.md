# PQ biometric evidence authentication

AI-based biometric evidence authentication for post-quantum digital identity systems (VIT B.Tech project).

A face-verification testbed where the evidence travels over a post-quantum channel
(ML-KEM-768 + ML-DSA-65), is bound to one transaction and one single-use challenge,
is checked by five attack detectors, and where **the detectors' verdict is itself
signed with ML-DSA-65** so it can be re-checked later. The same pipeline runs in a
classical mode (X25519 + Ed25519) to measure what post-quantum cryptography costs and
what it does and does not stop.

`CLAUDE.md` is the specification. `docs/RUNBOOK.md` has the step-by-step commands to
reproduce every result.

## Quick start (no weights, no Redis)

```bash
python -m venv .venv && . .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest -q                                          # 240+ tests, stub models, in-process Redis
python -m server --fakeredis                       # http://127.0.0.1:8000/health
```

In another terminal:

```bash
python -m client.enroll --user alice                # webcam photo (SPACE), keys for PQ and CLASSICAL
python -m client.capture --user alice --tx '{"type":"payment","amount":500,"currency":"INR","payee":"acct-42"}'
python scripts/verify_verdict.py --all              # every signed verdict re-checked offline
```

Without model weights every detector runs as a **stub** (fixed score, version `stub`).
`/health` and every signed verdict say so. To run the real models:

```bash
python scripts/fetch_models.py yunet landmarker buffalo_l        # face detector, M4 landmarker, ArcFace
pip install -r requirements-export.txt
python scripts/fetch_models.py minifasnet gend
python scripts/export_onnx.py m1 && python scripts/export_onnx.py m3
```

With Docker: `docker compose up` (Redis + server; weights mounted from `models/weights`).

## How a transaction works

1. `POST /challenge` returns a 32-byte nonce valid for 60 s, `tx_hash = SHA-384(transaction)`,
   two liveness actions derived from the nonce, and the server's KEM key (pinned at enrolment).
2. The client shows the actions and records 4 s of video plus camera metadata.
3. It signs `SHA-384(manifest)` (mode, challenge, nonce, tx_hash, video hash, time) with its
   device key, derives an AES-256-GCM key from an ML-KEM-768 encapsulation, and sends the sealed bundle.
4. The server checks, in order: challenge valid and unused (consumed atomically), mode not
   downgraded, decryption, nonce and transaction binding, video hash, device signature.
5. The matcher and M1–M5 run in parallel; any module over its threshold rejects, otherwise the matcher decides.
6. The server signs the verdict (every score, model versions, thresholds version, decision) and stores it.

| Module | Method | Rejects with |
|---|---|---|
| Matcher | ArcFace (InsightFace buffalo_l) cosine vs enrolled template | NO_MATCH |
| M1 presentation attack | MiniFASNet (Silent-Face-Anti-Spoofing), CelebA-Spoof | PAD_FAIL |
| M2 morphing | D-MAD: ArcFace(reference) − ArcFace(probe) → RBF SVM, FRLL-Morphs | MORPH_FLAG |
| M3 deepfake | GenD (CLIP ViT-L/14), OpenForensics | DEEPFAKE_FAIL |
| M4 replay | protocol checks + MediaPipe confirms the two actions in order | SCRIPT_FAIL |
| M5 injection | virtual-camera name rule + gradient boosting on frame timing and metadata | INJECTION_FAIL |

## Repository

```
crypto/         CryptoSuite (both modes), canonical manifest
server/         FastAPI app, challenge store, checks 1-6, decision, signed verdict, detector pipeline
client/         enrol and capture CLIs, protocol (sealing, key pinning)
preprocessing/  face detection + crops, video container
detectors/      matcher, m1_pad, m2_morph, m3_deepfake, m4_replay, m5_injection
eval/           metrics, crypto benchmark, perf, attack matrix, locust, plots; results/ holds the CSVs
scripts/        downloads, dataset filters, training, ONNX export, calibration, replay suite, verdict check
tests/          pytest for every module
```

## Datasets and licences

All datasets are downloaded by script (CelebA-Spoof by hand, behind its agreement) and
never committed: CelebA-Spoof (non-commercial research), FRLL (CC BY 4.0), FRLL-Morphs
(CC BY-SA 4.0), OpenForensics (CC BY 4.0), and the team's own M5 recordings (written
consent, `docs/consent.txt`). FRLL is London-based and CelebA-Spoof celebrity-heavy, so
performance on Indian faces is untested. InsightFace models are for non-commercial research only.
