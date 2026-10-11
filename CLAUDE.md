# CLAUDE.md — PQ Biometric Evidence Authentication (VIT B.Tech)

AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems.
Team: Abhijay (server, matcher, M5, integration, evaluation), Aviral (crypto, client, binding, M4), Krish (M1, M2, M3).
Timeline: 12 Oct – 15 Nov 2026. Supervisors: Dr. Aswani Kumar Cherukuri, Dr. Rajasekhara Reddy D.

Read this file fully before every task. Build only what is described here.

---

## 1. Goal

A working, reproducible prototype that:
1. Protects face evidence with ML-KEM-768 (key exchange) and ML-DSA-65 (signatures).
2. Detects five attack classes: presentation (M1), morphing (M2), deepfake (M3), replay (M4), injection (M5).
3. Binds each piece of evidence to one transaction and one challenge, so it cannot be reused.
4. Measures detection accuracy, latency, throughput, CPU, memory, and PQC overhead (classical vs PQ).
5. Signs the final verdict (scores + decision) so it can be re-checked later. **This is the project's novelty.**

## 2. Architecture

### Crypto modes (only these two)
| Mode | Key exchange | Signature | Symmetric |
|---|---|---|---|
| CLASSICAL | X25519 (ephemeral client, static server) | Ed25519 | HKDF-SHA384 → AES-256-GCM |
| PQ | ML-KEM-768 | ML-DSA-65 | HKDF-SHA384 → AES-256-GCM |

Library: `cryptography` >= 48 for everything. Mode comes from `config.yaml`.

### Enrolment
1. Client generates a device signing key pair (Ed25519 or ML-DSA-65) and sends the public key.
2. Server stores the public key and the ArcFace template of the reference photo (encrypted at rest).

### Challenge (`POST /challenge`)
Server creates and stores in Redis (TTL 60 s, status unused):
- `challenge_id`, `nonce` (32 random bytes)
- `tx_hash = SHA-384(canonical_json(tx_context))`
- `actions` = two of {blink_twice, turn_left, turn_right}, derived from HMAC(server_secret, nonce)
- `mode`, and the server KEM public key

### Capture client (Layer A)
1. Shows the two actions, records a 4-second webcam video.
2. Logs per-frame timestamps and camera metadata (device name, requested vs delivered resolution/FPS, resolution-change latency).
3. Builds `manifest = {mode, challenge_id, nonce, tx_hash, sample_hash = SHA-384(video), timestamp}` as canonical JSON (sorted keys, no spaces).

### PQ channel (Layer B)
1. `sig = sign(device_key, SHA-384(manifest))`
2. `(kem_ct, ss) = encapsulate(server_kem_pub)`
3. `K = HKDF-SHA384(ikm=ss, salt=nonce, info="BEA-v1|" + mode + "|" + challenge_id)`
4. `payload = AES-256-GCM(K, video || metadata || manifest || sig, AAD = mode + "|" + challenge_id)` (each part length-prefixed)
5. Send `{challenge_id, mode, kem_ct, iv, payload}` (length-prefixed, `SealedBundle.to_bytes()`) to `POST /verify`.

The client pins the server KEM key at enrolment and refuses a challenge that names another key or a different `tx_hash` than the transaction it shows.

### Verification server (Layer C) — check order is fixed
Stop at the first failure and return its reason code. The lookup runs first because the mode to compare against is stored with the challenge; a downgraded request therefore still consumes it.

| # | Check | Reason code |
|---|---|---|
| 1 | Challenge exists, unexpired, unused → mark used atomically (Redis GETDEL) | EXPIRED / REPLAY |
| 2 | Mode matches stored challenge | MODE_MISMATCH |
| 3 | Decapsulate, derive K, decrypt (GCM tag valid) | TAMPER |
| 4 | Nonce and tx_hash match issued values | TX_MISMATCH |
| 5 | Recomputed sample hash matches manifest | HASH_MISMATCH |
| 6 | Signature verifies under enrolled device key | SIG_INVALID |

Then run in parallel (asyncio + ONNX Runtime sessions loaded once at startup):

| Module | Method | Reason code |
|---|---|---|
| Matcher | InsightFace buffalo_l ArcFace, cosine similarity vs template | NO_MATCH |
| M1 PAD | MiniFASNetV2 (scale 2.7) + MiniFASNetV1SE (scale 4.0), the two released Silent-Face-Anti-Spoofing models, fine-tuned on CelebA-Spoof | PAD_FAIL |
| M2 Morph | ArcFace(enrolled ref) − ArcFace(live probe) → RBF SVM | MORPH_FLAG |
| M3 Deepfake | GenD (CLIP ViT-L/14), frame scores averaged | DEEPFAKE_FAIL |
| M4 Replay | Checks 1–6 + MediaPipe confirms the two actions appear in order | SCRIPT_FAIL |
| M5 Injection | Rules + HistGradientBoosting on capture metadata and frame timing | INJECTION_FAIL |

**Decision:** any failed check or module score above its threshold → reject. Otherwise accept if matcher similarity ≥ threshold. Per-module thresholds set at BPCER 0.5% on validation data (M2: 1%, see §4), stored in `config.yaml`.

**Signed verdict:** server signs `{challenge_id, tx_hash, sample_hash, scores{match, m1..m5}, model_versions, thresholds_version, decision, reason_code, timestamp}` with its own key (ML-DSA-65 in PQ mode) and stores it in SQLite: exactly one per consumed challenge, protocol rejections included. Only hashes, never face images.

**Stubs:** a module whose weights are missing runs as a stub (fixed score, version `stub`) when `detectors.allow_stubs` is true. The version is in every verdict and in `/health`, so a stubbed result can never pass as a real one.

**Client response:** only `accepted` / `rejected` + `challenge_id`. Reason codes go to the server log.

### Evaluation (Layer D)
- Detection: APCER (per attack type), BPCER (ISO/IEC 30107-3), EER, AUC.
- Performance: p50/p95 latency per stage, throughput (locust), CPU % and peak memory (psutil), bytes per request.
- PQC overhead: same pipeline in CLASSICAL vs PQ, detectors unchanged.
- Every stage records `time.perf_counter_ns()` into a per-request timing dict.

## 3. Out of scope — refuse if suggested
Hybrid crypto mode, device/hardware attestation, monotonic counters, TLS channel binding, CNSA 2.0 (ML-KEM-1024/ML-DSA-87), audit hash chains, second models per module, learned/logistic fusion, web frontend, dashboards, gated datasets (FaceForensics++, Celeb-DF, Idiap, SMDD).

## 4. Datasets (all open; never commit to git)
| Use | Dataset | Source |
|---|---|---|
| M1 (+ M4 screen replay) | CelebA-Spoof, faces ≥112 px, official split | github.com/Davidzhangyuanhan/CelebA-Spoof (Drive links in README) |
| M2 + matcher | FRLL (1350×1350, CC BY 4.0) | figshare.com/articles/dataset/Face_Research_Lab_London_Set/5047666 |
| M2 | FRLL-Morphs (CC BY-SA 4.0) | zenodo.org/record/4415159 |
| M3 | OpenForensics Val + Test-Dev (CC BY 4.0) | zenodo.org/records/5528418 |
| M5 | Own recordings: real webcam vs OBS Virtual Camera vs v4l2loopback | team members only, written consent |

Rules: split by identity (no person in both train and test). Read CelebA-Spoof label meanings from its metadata files; do not assume indices. Report results per attack type and per morph method, not one average.

**M2 protocol (changed 2026-10-11, approved):** FRLL has one neutral and one smiling front image per subject, so
fixed val/test halves gave only 20 bona fide pairs each. Instead:
- bona fide pairs in both directions: neutral_front → {smiling_front, four 3/4 views} and
  smiling_front → {neutral_front, four 3/4 views} (10 per subject, 1,020 in all);
- morph pairs: morph(A,B) → {smiling_front, four 3/4 views} of A and of B (same 3/4 probes, so pose cannot separate the classes);
- 5-fold cross-validation grouped by identity (GroupKFold over subjects): every subject is tested once; a morph is
  tested only in the fold holding both its subjects and trained on only when neither is in the test fold;
- threshold at **BPCER 1%** (not 0.5%) on the pooled out-of-fold bona fide scores;
- every M2 result states its exact number of bona fide pairs.

## 5. Repo layout
```
settings.py    config loader (config.yaml, PQBIO_CONFIG, PQBIO_REDIS_URL)
crypto/        suite.py, manifest.py
server/        app.py, challenge.py, checks.py, decision.py, verdict.py, pipeline.py, store.py, timing.py, __main__.py
client/        enroll.py, capture.py, protocol.py
preprocessing/ faces.py   (one detection → 112x112, 224x224, 80x80 crops), video.py (JPEG-frame container)
detectors/     base.py, matcher/ m1_pad/ m2_morph/ m3_deepfake/ m4_replay/ m5_injection/
eval/          metrics.py, harness.py, perf.py, crypto_bench.py, attack_matrix.py, locustfile.py, plots.py, results/
scripts/       fetch_models.py, download_*.py, filter_celeba_spoof.py, record_m5.py, replay_attacks.py, calibrate.py,
               verify_verdict.py, eval_detectors.py, eval_matcher.py, train_m1.py, train_m2.py, train_m5.py, export_onnx.py
tests/         pytest for every module
data/          (gitignored)
models/weights (gitignored)
docs/
config.yaml  requirements.txt  Dockerfile  docker-compose.yml  README.md
```

## 6. Conventions
- Python 3.11, type hints, pytest for every module. No notebooks in main code.
- No hard-coded paths; everything in `config.yaml`.
- Develop and benchmark on Linux / WSL2 / Docker.
- Never invent results; every number in docs must come from a CSV in `eval/results/`.
- Small commits, one step per branch, merge via pull request.

## 7. Commands
```
pip install -r requirements.txt
docker run -p 6379:6379 redis:7
uvicorn server.app:app --port 8000          # or: python -m server --fakeredis   (no Redis install)
pytest -q
docker compose up
```

---

## 8. Implementation plan

Mark each item `[x]` when its **Done when** check passes. Work on one step per session.

**Rules:** tick `[x]` only when the full *Done when* check passed with real models and real data.
Steps with code but no real-model run go in the progress log as "code done, not verified".
No stub numbers in `eval/results/` or `docs/`. Commands per step: `docs/RUNBOOK.md`.

**Progress log**
- 2026-10-11 S0, S1, S2: done (their *Done when* checks need no models: pip/pytest; NIST vectors; fakeredis reason-code tests).
- 2026-10-11 S3: done. ArcFace buffalo_l (w600k_r50) + YuNet on FRLL: 102 subjects enrolled (neutral_front), 501/510 probes with a face (9 three-quarter views without a face >= 112 px, excluded). Threshold at FMR 0.1% on the val half = 0.2192; FNMR 0 on val and test (eval/results/matcher.csv). Server enrol/verify checked with the real matcher. config.yaml `thresholds.match` is still the uncalibrated 0.35 until S11 runs calibrate.py.
- 2026-10-11 S10: done for the pretrained models. MiniFASNetV2 / V1SE (Silent-Face @ b6d5f04) and GenD CLIP-L/14 exported
  (opset 17); PyTorch vs ONNX cosine min 1.000000 on 200 real face crops each (FRLL patches for M1, OpenForensics Val
  for M3); CPU 3.8 / 3.9 / 616 ms per input (eval/results/onnx_export.csv). Fine-tuned M1 must be re-exported after S7.
- 2026-10-11 GenD: its own loader fails under transformers 5.19 (from_pretrained inside a meta-device init), so it is
  exported from a separate venv with transformers 4.56.2 (`.venv-gend456`, export only). The server runs the ONNX file.
  ONNX + server preprocessing vs GenD's own pipeline (CPU): max |dP| 5.2e-6 on 50 faces; GenD on GPU differs from CPU
  by up to 4.4e-3 (TF32) with the same class (eval/results/m3_gend_equivalence.csv).
- 2026-10-11 Server: matcher, M1 (pretrained), M3, M4 real; M2 stub until S7 saves its model; M5 rules only.
  GenD scores 3 of the sampled frames per request (`detectors.m3_deepfake.max_frames`, CPU cost). `/health` lists stubs.
- 2026-10-11 Webcam works again (status OK, 640x480 @ 30 fps). S6 live check needs a team member at the camera.
- 2026-10-11 S5 (M3 part): zero-shot GenD ONNX on OpenForensics Val, 3,951 real / 9,861 fake faces >= 112 px: AUC 0.930,
  EER 0.141; threshold at BPCER 0.5% set on these same Val real faces = 0.9754 -> APCER 0.700 (eval/results/m3_deepfake_zero_shot.csv).
  Test-Dev (independent) to be added. M1 zero-shot waits for CelebA-Spoof.
- 2026-10-11 S7 (M2 part): D-MAD SVM, 5-fold GroupKFold, pooled out-of-fold: 1,002 bona fide pairs, threshold 0.4908 at
  BPCER 1%; EER 0.0039, AUC 0.9999; APCER amsl 0.0038 (3,984 pairs), facemorpher 0.0022 (2,265), opencv 0.0022 (2,265),
  stylegan 0.0000 (775), webmorph 0.0026 (2,265) (eval/results/m2_morph.csv). Caveat: YuNet (score 0.8) finds no face in
  most StyleGAN morphs (226/300 sampled; all found at 0.5), so StyleGAN results cover only the confidently detected ~30%;
  852 of 7,663 images had no face (9 FRLL 3/4 views, the rest morphs). Threshold is from the pooled out-of-fold scores
  it is reported on (no separate validation set). config.yaml thresholds stay uncalibrated until S11.
- 2026-10-11 S4, S5 (M1), S6, S7 (M1), S8, S9, S11, S12, S13: code done, not verified.
- 2026-10-11 S9 replay suite was run with stub detectors and synthetic video; that CSV was removed (rule above). Rerun with real models.

### Week 1 (12–18 Oct) — milestone: one genuine PQ transaction end to end
- [x] **S0 Repo skeleton** (Abhijay) — layout, requirements.txt, config.yaml, .gitignore, README.
  Done when: `pip install -r requirements.txt` and `pytest -q` run.
- [x] **S1 Crypto suite** (Aviral) — `CryptoSuite` with both modes: KEM keygen, encapsulate/decapsulate, sign/verify, HKDF, AES-GCM; canonical manifest + SHA-384.
  Done when: round-trip tests pass in both modes; flipped byte / wrong key / changed manifest fail; ML-KEM-768 ciphertext = 1,088 B, ML-DSA-65 signature = 3,309 B; NIST known-answer vectors (usnistgov/ACVP-Server) pass.
- [x] **S2 Server skeleton** (Abhijay) — `/enroll`, `/challenge`, `/verify`; Redis store; checks 1–6 in order; detector stubs return 0.0; stage timers.
  Done when: a pytest (with fakeredis) exists and passes for every reason code plus a valid request.
- [x] **S3 Matcher** (Abhijay) — `embed()`, `similarity()`; enrol FRLL neutral_front, probe smiling/3-4 views; wire into `/enroll` and `/verify`.
  Done when: `eval/results/matcher.csv` has FMR, FNMR and threshold at FMR 0.1%.
- [ ] **S4 Datasets + preprocessing** (Krish) — download scripts; `faces.py` crops; CelebA-Spoof filter CSV.
  Done when: CSVs exist, a 20-crop sample grid looks correct, no identity overlap.
- [ ] **S5 Baselines + metrics** (Krish) — wrap MiniFASNet (M1) and GenD (M3); `eval/metrics.py` (APCER, BPCER, EER, AUC); zero-shot runs.
  Done when: zero-shot M1 and M3 results saved.

### Week 2 (19–25 Oct) — milestone: each piece works on its own
- [ ] **S6 Capture client** (Aviral) — `enroll.py`, `capture.py`: challenge → actions → 4 s video → metadata → manifest → sign → encrypt → verify.
  Done when: a team member enrols and is accepted in both modes.
- [ ] **S7 M1/M2/M3** (Krish) — fine-tune MiniFASNet; build D-MAD SVM on FRLL-Morphs (morph-ref vs genuine-ref pairs, identity split); evaluate GenD zero-shot (the optional LayerNorm-only tuning is dropped, decision 2026-10-11).
  Done when: per-attack-type results CSV per module, including APCER on screen-replay types (PC/Pad/Phone) and per morph method.
- [ ] **S8 M5 data + features** (Abhijay) — `record_m5.py` (bona_fide / obs / v4l2loopback, ~40–60 sessions each); `features.py`.
  Done when: feature CSV covers all three classes.

### Week 3 (26 Oct – 1 Nov) — milestone: full pipeline in both modes, demo-ready
- [ ] **S9 M4 + replay suite** (Aviral) — MediaPipe blink/yaw action check; `replay_attacks.py`: exact re-submit, expired challenge, old nonce, swapped tx_context, flipped byte (1,000 each) + 1,000 fresh genuine runs.
  Done when: 100% of attacks rejected with correct code; 0% false rejection from protocol checks.
- [x] **S10 ONNX export** (Krish) — MiniFASNet and GenD to ONNX (opset 17).
  Done when: PyTorch vs ONNX cosine > 0.999 on 200 images; CPU latency recorded.
- [ ] **S11 Integration** (Abhijay) — real modules in parallel; M5 classifier (leave-one-tool-out); `calibrate.py` thresholds; decision rule; signed verdict + `verify_verdict.py`; Dockerfile + docker-compose.
  Done when: `docker compose up` works; genuine accepted; print, screen replay, OBS and morphed enrolment each rejected with the right code; editing a stored score makes `verify_verdict.py` fail.

### Week 4 (2–8 Nov) — milestone: all results collected
- [ ] **S12 Evaluation** (all; Abhijay leads) — `perf.py` (500 runs/mode, p50/p95 per stage, bytes), locust throughput, psutil CPU/memory, `crypto_bench.py` (1,000 runs/op/mode), `attack_matrix.py` (rejection rate per attack per mode), `plots.py`.
  Done when: every metric in the requirements mapping has a CSV in `eval/results/`.

### Week 5 (9–15 Nov) — milestone: final report and demo
- [ ] **S13 Analysis + reproducibility** (all) — `docs/RESULTS.md` from CSVs only; README with full reproduction order; fresh-clone rebuild by a teammate; demo script (enrol → accept → print reject → replay reject → OBS reject → replayed bundle reject → verify verdict).
  Done when: a teammate reproduces the results from a fresh clone.

### If behind schedule
Use pretrained MiniFASNet and GenD without fine-tuning. Present at the Week 3 milestone with replay results and crypto benchmarks as the finished evaluations.

## 9. Requirement mapping (keep in sync)
| Faculty requirement | Component | Evidence |
|---|---|---|
| Testbed combining PQC + AI verification | Layers A–D, docker-compose | Genuine accept in both modes |
| ML-KEM + ML-DSA integration | crypto/suite.py | Tests + NIST vectors |
| AI detection of 5 attack classes | M1–M5 | Per-module results CSVs |
| Accuracy, latency, throughput, CPU, memory | eval/ | perf, locust, psutil CSVs |
| Transaction binding + freshness | Checks 1–6 + liveness actions | replay.csv (100% rejection) |
| PQC overhead | crypto_bench + perf, two modes | Δ ms, Δ bytes, share of total |
| Security–performance trade-off | attack_matrix + RESULTS.md | Attack × mode table |
| Reproducible prototype | README, pinned versions, Docker | Fresh-clone rebuild |
