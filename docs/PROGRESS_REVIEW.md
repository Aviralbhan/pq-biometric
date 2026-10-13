# Progress review (2026-10-13)

AI-based biometric evidence authentication for post-quantum digital identity systems.
Numbers below are generated from `eval/results/*.csv` by `scripts/progress_review.py`.
Live demo: `docs/DEMO.md`. Thresholds in use: `preliminary-val-2026-10-12-10537d5e` (preliminary, calibrated on validation data).

## Status

| step | status | note |
|---|---|---|
| S0 Repo, config, CI, Docker files | done | Docker image not built yet (no Docker on the dev laptop) |
| S1 Crypto suite (ML-KEM-768, ML-DSA-65; X25519, Ed25519) | done | NIST ACVP vectors pass |
| S2 Server and checks 1-6 | done | every reason code tested in both modes |
| S3 Matcher (ArcFace) on FRLL | done | table below |
| S4 Datasets and preprocessing | partly | FRLL, FRLL-Morphs, OpenForensics in; CelebA-Spoof not yet |
| S5 Zero-shot M1 / M3 | partly | M3 done (Val; Test-Dev overnight); M1 waits for CelebA-Spoof |
| S6 Capture client | partly | protocol verified live in both modes; acceptance under final thresholds pending |
| S7 M1 / M2 / M3 training | partly | M2 done; M3 LayerNorm-tuned; M1 fine-tune waits for CelebA-Spoof |
| S8 M5 recordings | partly | 10 bona fide sessions; OBS sessions need OBS installed |
| S9 M4 and replay suite | partly | code done, live M4 works; 1,000-run suite with real models pending |
| S10 ONNX export | done | PyTorch vs ONNX parity table below |
| S11 Integration and calibration | partly | real models in the server; thresholds preliminary |
| S12 Evaluation (latency, throughput, attack matrix) | pending | crypto micro-benchmark only |
| S13 Report and reproducibility | pending |  |

## Results

**Matcher** (ArcFace buffalo_l on FRLL, 102 subjects, halves by identity):

| half | threshold (FMR 0.1% on val) | FMR | FNMR | genuine / impostor pairs |
|---|---|---|---|---|
| val | 0.2192 | 0.09% | 0.00% | 255 / 12750 |
| test | 0.2192 | 0.10% | 0.00% | 255 / 12750 |

**M2 morph detection** (D-MAD SVM on FRLL + FRLL-Morphs, 5-fold identity-grouped CV):

| BPCER target | BPCER achieved | APCER amsl | APCER facemorpher | APCER opencv | APCER stylegan | APCER webmorph |
|---|---|---|---|---|---|---|
| 0.50% | 0.59% | 0.37% | 0.22% | 0.26% | 0.00% | 0.35% |
| 1.00% | 1.37% | 0.27% | 0.22% | 0.13% | 0.00% | 0.22% |
| 5.00% | 5.29% | 0.12% | 0.00% | 0.00% | 0.00% | 0.09% |

Pooled held-out over **1020 bona fide pairs**; morph pairs: amsl 4030, facemorpher 2290, opencv 2290, stylegan 2290, webmorph 2290. EER 0.31%, AUC 0.9999. Images with no face found: amsl 0, facemorpher 0, opencv 0, stylegan 0, webmorph 0. Thresholds per fold from an inner identity-grouped CV on training subjects only.

**M3 deepfake detection** (GenD CLIP ViT-L/14, OpenForensics):

Zero-shot GenD, thresholds from all of Val. The Val rows are that calibration set itself (BPCER matches its target by construction); the Test-Dev rows, when present, are the headline independent result:

| split | BPCER target | threshold | BPCER | APCER | EER | AUC | real / fake faces |
|---|---|---|---|---|---|---|---|
| test-dev | 0.50% | 0.9754 | 0.37% | 68.06% | 12.61% | 0.9434 | 10468 / 19865 |
| test-dev | 1.00% | 0.9696 | 0.56% | 60.04% | 12.61% | 0.9434 | 10468 / 19865 |
| test-dev | 5.00% | 0.9229 | 4.00% | 28.32% | 12.61% | 0.9434 | 10468 / 19865 |
| val | 0.50% | 0.9754 | 0.48% | 70.01% | 14.10% | 0.9299 | 3951 / 9861 |
| val | 1.00% | 0.9696 | 0.99% | 61.94% | 14.10% | 0.9299 | 3951 / 9861 |
| val | 5.00% | 0.9229 | 4.99% | 30.68% | 14.10% | 0.9299 | 3951 / 9861 |

Test-Dev, zero-shot vs LayerNorm-tuned (both with thresholds from the same 20% Val calibration split):

| model | thresholds from | BPCER target | BPCER | APCER | EER | AUC |
|---|---|---|---|---|---|---|
| zero-shot | Val calib 20% | 0.50% | 0.43% | 65.16% | 12.61% | 0.9434 |
| zero-shot | Val calib 20% | 1.00% | 0.90% | 53.42% | 12.61% | 0.9434 |
| zero-shot | Val calib 20% | 5.00% | 5.26% | 24.20% | 12.61% | 0.9434 |
| LayerNorm-tuned | Val calib 20% | 0.50% | 0.21% | 11.55% | 1.75% | 0.9978 |
| LayerNorm-tuned | Val calib 20% | 1.00% | 1.39% | 2.37% | 1.75% | 0.9978 |
| LayerNorm-tuned | Val calib 20% | 5.00% | 5.06% | 0.52% | 1.75% | 0.9978 |

LayerNorm-only tuning on 80% of Val, measured on the 20% calibration split: epoch 0: AUC 0.9354, EER 13.15%; epoch 1: AUC 0.9885, EER 4.18%; epoch 2: AUC 0.9965, EER 1.56%.

**Replay suite** (protocol attacks, both modes):

_pending: the 1,000-run suite with real models is S9 (the earlier stub run was discarded)._

**Crypto cost** (sizes and median times per operation):

|  | X25519 / Ed25519 | ML-KEM-768 / ML-DSA-65 |
|---|---|---|
| KEM public key (B) | 32 | 1184 |
| KEM ciphertext (B) | 32 | 1088 |
| signature (B) | 64 | 3309 |
| extra per request (B) | - | 4301 |

| operation (median of 1000 runs) | classical (us) | PQ (us) |
|---|---|---|
| encapsulate | 48.1 | 38.8 |
| decapsulate | 22.9 | 40.2 |
| sign_manifest | 22.9 | 473.1 |
| verify_manifest | 63.6 | 111.1 |
| sign_verdict | 33.0 | 468.35 |
| seal_bundle | 1536.4 | 1526.8 |
| unseal_bundle | 1076.3 | 1090.0 |

Machine: Windows 11, cryptography 50.0.2, payload 2048 KB.

**ONNX export parity** (the server runs these files):

| model | parity inputs | cosine min (PyTorch vs ONNX) | pass |
|---|---|---|---|
| gend_clip_l14.onnx | face crops listed in m3_val.csv | 1.000000 | True |
| 2.7_80x80_MiniFASNetV2.onnx | face crops listed in m1_parity_frll.csv (FRLL) | 1.000000 | True |
| 4_0_0_80x80_MiniFASNetV1SE.onnx | face crops listed in m1_parity_frll.csv (FRLL) | 1.000000 | True |
| gend_clip_l14_ln.onnx | face crops listed in m3_val_calib.csv | 1.000000 | True |

## Known limitations

- M1 runs the pretrained MiniFASNet with its own argmax rule: no CelebA-Spoof data yet, so no M1 numbers.
- M5 runs only its virtual-camera name rule; the classifier needs ~40-60 sessions per class (have 10 bona fide, 0 OBS).
- Thresholds are preliminary (validation data only, never webcam captures); final calibration is S11.
- FRLL is studio-quality and London-based, CelebA-Spoof celebrity-heavy: performance on Indian faces and webcams is untested.
- M2 thresholds come from cross-validation on FRLL-Morphs (no separate morph test set); morph generators outside FRLL-Morphs are untested.
- Runs on Windows with an in-process Redis (fakeredis); the Docker/Linux setup and latency measurements are S12 work.

## Next steps

| when | what |
|---|---|
| by 18 Oct | CelebA-Spoof: M1 zero-shot at three operating points, then fine-tuning and re-export (S5, S7, S10) |
| by 20 Oct | Replay suite with real models, 1,000 runs per attack per mode (S9) |
| 19-25 Oct | M5 recordings across the team: webcam, OBS video, OBS image (S8) |
| by 1 Nov | Final calibration on validation data, M5 classifier, Docker build on Linux (S11) |
| 2-8 Nov | Latency per stage, throughput, CPU/memory, attack x mode matrix on an idle Linux machine (S12) |
| 9-15 Nov | Report from the CSVs, fresh-clone rebuild by a teammate, final demo (S13) |
