# Results so far, what they mean, and what is left

Status on 12 Oct 2026, before the progress review. Every number below comes from a CSV in
`eval/results/` (named under each table). Results that are still running or not started are marked
**pending**; nothing is estimated. Architecture and formulas: [ARCHITECTURE.md](ARCHITECTURE.md).

Thresholds in use are **preliminary** (version `preliminary-val-2026-10-12-10537d5e`): set on
validation data only, never on webcam captures. Final calibration is step S11.

---

## 1. Summary in plain words

| Part | Result | Meaning |
|---|---|---|
| Face matcher | 0 genuine pairs rejected and about 1 in 1,000 impostor pairs accepted on held-out subjects | It recognises the right person very reliably on good-quality photos |
| Morph detection (M2) | Fewer than 4 in 1,000 morphs pass when about 6 in 1,000 genuine users are rejected | Morphed enrolment photos from five different morphing tools are almost always caught |
| Deepfake detection (M3), off the shelf | Separates real from fake well (AUC 0.93) but lets 70% of fakes through at the strict setting | The published model needs adapting to this kind of data |
| Deepfake detection (M3), tuned | On held-out validation faces: AUC 0.997, equal error 1.6% (was 13.1%) | Small tuning helps a lot; the independent Test-Dev check is still running |
| Live system on the webcam | Genuine user accepted in both crypto modes; the exported models give the same answers as the originals | The full pipeline works end to end on real hardware |
| Crypto overhead, replay suite, M1, M5 | **pending** | Measured in the next weeks (section 4) |

---

## 2. How the results were produced

| What | Data | Method |
|---|---|---|
| Matcher | FRLL (102 people, London face set): enrol a neutral front photo, probe smiling and four 3/4-view photos | People split in two halves; threshold set on one half (validation) at 0.1% false matches, reported on both |
| M2 morph | FRLL + FRLL-Morphs (AMSL, FaceMorpher, OpenCV, StyleGAN, WebMorph) | 5-fold cross-validation grouped by person, so nobody is in training and testing at once; thresholds from the training people only (inner cross-validation) |
| M3 deepfake | OpenForensics Val (3,951 real / 9,861 fake faces) and Test-Dev | Zero-shot GenD; then LayerNorm-only tuning on 80% of Val (split by source image), threshold from the other 20% |
| Live webcam | Aviral at the laptop webcam, real models, preliminary thresholds | Genuine captures in both modes; every capture is logged, including rejections |
| Model export | The ONNX files the server runs | Same inputs through PyTorch and ONNX; outputs compared |

Every detector is reported at three operating points: the share of attacks accepted (APCER) when
0.5%, 1% and 5% of genuine users are rejected (BPCER), plus the equal error rate (EER) and AUC.

---

## 3. Results and what they mean

### 3.1 Face matcher (ArcFace) — `matcher.csv`

| Half of the people | Threshold | False match rate (FMR) | False non-match rate (FNMR) | Genuine pairs | Impostor pairs |
|---|---|---|---|---|---|
| validation | 0.2192 | 0.094% | 0.0% | 255 | 12,750 |
| test | 0.2192 | 0.102% | 0.0% | 255 | 12,750 |

**Meaning.** The threshold chosen on the validation half (0.1% false matches) gives the same false
match rate on people it has never seen, and no genuine person is rejected, even with smiling and
3/4-view photos. No image failed face detection. **Caution:** FRLL photos are studio quality; webcam
images are harder (section 3.5 shows this).

### 3.2 Morph detection (M2, D-MAD) — `m2_morph.csv`, `m2_deploy_thresholds.csv`

Held-out results pooled over the 5 folds, **1,020 genuine pairs**:

| Genuine users rejected: target → achieved | AMSL (4,030 pairs) | FaceMorpher (2,290) | OpenCV (2,290) | StyleGAN (2,290) | WebMorph (2,290) |
|---|---|---|---|---|---|
| 0.5% → 0.59% | 0.37% | 0.22% | 0.26% | 0.00% | 0.35% |
| 1% → 1.37% | 0.27% | 0.22% | 0.13% | 0.00% | 0.22% |
| 5% → 5.29% | 0.12% | 0.00% | 0.00% | 0.00% | 0.09% |

Cells are APCER: the share of morph attacks wrongly accepted. Equal error rate 0.31%, AUC 0.9999.
No image failed face detection (`m2_images_without_face.csv`). Deployed threshold (0.5%): **0.5608**.

**Meaning.** When the system rejects about 6 in 1,000 genuine users, fewer than 4 in 1,000 morph
attacks get through, for all five morphing tools. The thresholds were set only from training people,
so these numbers are not optimistic. **Caution:** all morphs come from one dataset (FRLL-Morphs);
other morphing tools are untested. An earlier way of setting thresholds (from the SVM's own training
scores) failed, rejecting 37% of genuine users (`m2_morph_train_score_thresholds.csv`), and was
replaced by the nested cross-validation above.

### 3.3 Deepfake detection (M3, GenD) — `m3_deepfake_zero_shot.csv`, `m3_ln_tuning.csv`

**Zero-shot (the published model, unchanged)** on OpenForensics Val (3,951 real / 9,861 fake faces).
The thresholds are set on these same faces, so the genuine-rejection column matches its target by construction:

| Target genuine rejection | Threshold | Achieved BPCER | Fakes accepted (APCER) | EER | AUC |
|---|---|---|---|---|---|
| 0.5% | 0.9754 | 0.48% | 70.0% | 14.1% | 0.930 |
| 1% | 0.9696 | 0.99% | 61.9% | 14.1% | 0.930 |
| 5% | 0.9229 | 4.99% | 30.7% | 14.1% | 0.930 |

**LayerNorm-only tuning** (0.03% of the weights trained), measured on the held-out 20% of Val:

| Epoch | AUC | EER |
|---|---|---|
| 0 (zero-shot) | 0.935 | 13.1% |
| 1 | 0.989 | 4.2% |
| 2 | 0.997 | 1.6% |

**Meaning.** Off the shelf, GenD ranks real and fake faces well (AUC 0.93), but OpenForensics fakes
look different from what it was trained on. At a strict setting (few genuine rejections) most fakes
pass. Tuning only the LayerNorm layers for two epochs cuts the equal error from 13.1% to 1.6% on
held-out faces. **Caution:** the tuning and calibration faces come from the same dataset as the
tuning data. The independent check, **OpenForensics Test-Dev (10,468 real / 19,865 fake faces), is
still running: pending.** The live system currently uses the tuned model.

### 3.4 Exported models match the originals — `onnx_export.csv`, `m3_gend_equivalence.csv`

| Model the server runs | Check on 200 real faces | Result |
|---|---|---|
| MiniFASNetV2, MiniFASNetV1SE (M1) | PyTorch vs ONNX output similarity (cosine, minimum) | 1.000000, pass |
| GenD zero-shot and tuned (M3) | same | 1.000000, pass |

GenD's own pipeline vs our server path (ONNX + our preprocessing), 50 faces: largest difference in
P(fake) 0.0000052, same class on 50/50. **Meaning:** the server scores exactly what the published
models score; the numbers in 3.3 apply to the running system.

### 3.5 Live webcam test with real models — `genuine_webcam_captures.csv`

All 32 genuine captures of Aviral, every one logged:

| Phase | Mode | Accepted | Rejected (reason) |
|---|---|---|---|
| First live check (uncalibrated thresholds) | PQ / CLASSICAL | 0 / 1 | PQ: 1 (deepfake score above the then-uncalibrated 0.5) |
| After lighting and position changed, old enrolment | PQ | 0 | 10 (MORPH_FLAG: face no longer matched the enrolment photo) |
| Re-enrolled, zero-shot M3 | PQ / CLASSICAL | 5 / 4 | CLASSICAL: 1 (SCRIPT_FAIL: actions not seen in order) |
| Re-enrolled, tuned M3 | PQ / CLASSICAL | 4 / 5 | PQ: 1 (MORPH_FLAG, with a low face match) |

Scores of the 20 captures after re-enrolment: face match 0.79–0.88 on every accepted capture (threshold 0.2192),
deepfake score at most 0.35 (threshold 0.9182 tuned / 0.9754 zero-shot), photo-attack score at most 0.022 (threshold 0.5).

**Meaning.** With an enrolment made in the same conditions, the real user is accepted 18 times in 20
in both crypto modes, and with the preliminary thresholds the deepfake detector never rejects the real user
(it did once at the earlier, uncalibrated threshold of 0.5). The biggest practical
risk is a **change of lighting or position since enrolment**: then the face does not match, and the
system rejects (safely, but annoyingly). The demo therefore enrols at the start.

### 3.6 Security protocol and cryptography (tests, not CSVs)

- NIST ACVP known-answer tests for ML-KEM-768 and ML-DSA-65: all pass.
- Every server reason code (EXPIRED, REPLAY, MODE_MISMATCH, TAMPER, TX_MISMATCH, HASH_MISMATCH,
  SIG_INVALID and the six detector codes) is triggered and checked in both modes by the test suite.
- The demo shows replay, bit-flip tampering and transaction swapping rejected, and an edited
  verdict failing its signature check.

**Meaning.** The protocol behaves as designed. **The measured 1,000-run replay suite with the real
models and the crypto time/size table are pending** (section 4).

---

## 4. What is left

| Step | Left to do | Needs |
|---|---|---|
| S5 / S7 M3 | Finish the Test-Dev scoring (running), compare zero-shot vs tuned, choose the deployed model | Running overnight |
| S5 / S7 / S10 M1 | CelebA-Spoof: zero-shot results, fine-tuning, re-export, threshold | CelebA-Spoof download (manual, Google Drive) |
| S8 M5 | 40–60 sessions per class across the team: real webcam, OBS playing a video, OBS showing a photo; train and test the classifier (leave one tool out) | OBS Studio installed; team members with written consent |
| S9 | Replay suite with real models: 5 attacks × 1,000 runs × 2 modes, plus genuine runs | Idle machine |
| S11 | Final calibration on validation data; M5 classifier in the server; Docker image built and run on Linux | Steps above |
| S12 | Crypto cost table (running tonight), per-stage latency, throughput, CPU and memory in PQ vs classical on an idle machine, attack × mode matrix | Idle machine, attack datasets |
| S13 | Final report from the CSVs, a teammate rebuilds everything from a fresh clone, final demo | All of the above |

Known limitations to state in the report: FRLL is London-based and CelebA-Spoof celebrity-heavy, so
performance on Indian faces and real webcams is untested; morph and deepfake results come from one
dataset each; strong attackers on a rooted device and high-quality 3D masks are out of scope.

---

## 5. What the final review will show

1. **Live demo with every module final**: enrol → genuine accept in both modes → photo attack
   rejected (M1 + M4) → replay, tampering and transaction swap rejected → morph and deepfake
   rejected → **OBS virtual-camera injection rejected (M5)** → the signed verdict re-checked and
   an edited copy detected.
2. **Detection results** at BPCER 0.5%, 1% and 5% for every module: matcher (done), M1 on
   CelebA-Spoof by attack type, M2 by morph tool (done), M3 on Test-Dev zero-shot vs tuned, M5 by
   held-out tool.
3. **Security results**: the 1,000-run replay suite (every attack rejected with the right reason,
   no genuine user rejected by the protocol checks) and the attack × mode matrix showing which
   check or module stops each attack in PQ and classical mode.
4. **Cost of post-quantum cryptography**: time per operation, extra bytes per request,
   end-to-end latency per stage, throughput, CPU and memory, PQ vs classical, and the share of the
   total time spent on cryptography.
5. **Reproducibility**: Docker on Linux, and a teammate reproducing the results from a fresh clone.
