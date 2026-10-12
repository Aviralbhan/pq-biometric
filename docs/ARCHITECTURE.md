# Architecture, algorithms and attack countermeasures

AI-Based Biometric Evidence Authentication for Post-Quantum Digital Identity Systems.

This document explains **what the system is made of**, **which algorithms it uses** (besides
ML-KEM and ML-DSA), **the formulas behind them**, and **how each attack is stopped**. Every
value below is the one in the code and in `config.yaml` (thresholds are the preliminary ones,
version `preliminary-val-2026-10-12-10537d5e`).

---

## 1. Architecture at a glance

```
 ┌─────────────────────┐   1. POST /challenge   ┌──────────────────────────────────────────────┐
 │  Layer A: Capture   │ ─────────────────────► │  Layer C: Verification server (FastAPI)      │
 │  client (laptop)    │ ◄───────────────────── │                                              │
 │                     │  nonce, tx_hash,       │  Challenge store (Redis, 60 s, single use)   │
 │  webcam, 4 s video, │  2 actions, KEM key    │                                              │
 │  camera metadata    │                        │  Checks 1-6 (protocol)                       │
 │                     │   2. POST /verify      │     │                                        │
 │  Layer B: PQ        │ ─────────────────────► │     ▼                                        │
 │  channel            │  sealed bundle         │  Face detection + crops (YuNet)              │
 │  sign  (ML-DSA-65)  │  {challenge_id, mode,  │     │                                        │
 │  KEM   (ML-KEM-768) │   kem_ct, iv, payload} │     ▼  in parallel                            │
 │  AES-256-GCM        │                        │  Matcher │ M1 │ M2 │ M3 │ M4 │ M5             │
 └─────────────────────┘                        │     │                                        │
                                                │     ▼                                        │
         client only ever sees                  │  Decision rule ──► signed verdict (ML-DSA)   │
         "accepted" / "rejected"  ◄──────────── │                    stored in SQLite          │
                                                └──────────────────────────────────────────────┘
                                                 Layer D: evaluation harness (eval/) measures
                                                 accuracy, latency, bytes, PQ vs classical
```

| Layer | Code | What it does |
|---|---|---|
| A. Capture client | `client/capture.py`, `client/enroll.py` | Shows the two random actions, records a 4 s webcam video (JPEG frames), logs camera metadata and frame timestamps, builds the manifest |
| B. Post-quantum channel | `crypto/suite.py`, `crypto/manifest.py`, `client/protocol.py` | Signs the manifest, agrees a key, encrypts the evidence, pins the server key |
| C. Verification server | `server/` | Single-use challenges, checks 1–6, face pipeline, detectors M1–M5, decision, signed verdict |
| D. Evaluation | `eval/`, `scripts/` | Metrics (APCER, BPCER, EER, AUC, FMR, FNMR), crypto benchmark, replay suite, attack matrix |

The same code runs in two **crypto modes** so the cost of post-quantum cryptography can be measured:

| Mode | Key agreement | Signature | Symmetric |
|---|---|---|---|
| PQ | ML-KEM-768 (FIPS 203) | ML-DSA-65 (FIPS 204) | HKDF-SHA384 → AES-256-GCM |
| CLASSICAL | X25519 (ephemeral client, static server) | Ed25519 | HKDF-SHA384 → AES-256-GCM |

---

## 2. One transaction, step by step

1. **Challenge.** The server creates a 32-byte random nonce, the transaction hash and two
   liveness actions, stores them in Redis for 60 s, and sends them with its KEM public key.
2. **Capture.** The client checks the KEM key equals the one pinned at enrolment and that the
   transaction hash matches the transaction it is showing, then records 4 s of video while the
   user does the two actions.
3. **Manifest.** `manifest = {mode, challenge_id, nonce, tx_hash, sample_hash = SHA-384(video), timestamp}`,
   encoded as canonical JSON (sorted keys, no spaces, UTF-8).
4. **Sign, agree a key, encrypt** (formulas in §3.2).
5. **Server checks 1–6** in a fixed order, stopping at the first failure (§4).
6. **Detectors.** Faces are detected once; the matcher and M1–M5 run in parallel (§3.3).
7. **Decision and signed verdict.** The decision rule (§3.4) gives accepted / rejected; the server
   signs a record of every score and stores it. The client only sees accepted / rejected.

---

## 3. Algorithms (besides ML-KEM and ML-DSA) and their formulas

### 3.1 Hashing and encoding

**SHA-384** (FIPS 180-4) is used for every digest:

$$\text{tx\_hash} = \text{SHA-384}(\text{canonical\_json}(\text{tx\_context}))$$
$$\text{sample\_hash} = \text{SHA-384}(\text{video bytes}), \qquad d_{\text{manifest}} = \text{SHA-384}(\text{canonical\_json}(\text{manifest}))$$

**Canonical JSON** (sorted keys, separators `,` and `:`, UTF-8, no NaN) guarantees client and
server hash exactly the same bytes; the server rejects a manifest that is not in canonical form.

**Length-prefixed framing.** The encrypted plaintext is
`len(video)‖video ‖ len(metadata)‖metadata ‖ len(manifest)‖manifest ‖ len(sig)‖sig`
with 4-byte big-endian lengths, so the parts cannot be shifted into each other.

### 3.2 Post-quantum channel (Layer B)

**Signature on the manifest** (device key, ML-DSA-65 or Ed25519):

$$\sigma = \text{Sign}_{sk_{\text{device}}}\big(d_{\text{manifest}}\big)$$

**Key agreement.** PQ: $(c, ss) = \text{ML-KEM.Encaps}(pk_{\text{server}})$.
Classical: an ephemeral X25519 key pair $(e, E = e\cdot G)$, $ss = \text{X25519}(e, pk_{\text{server}})$, and the
"ciphertext" sent is $c = E$. Both give a 32-byte shared secret $ss$.

**HKDF-SHA384** (RFC 5869) turns $ss$ into the AES key, bound to this challenge and mode:

$$PRK = \text{HMAC-SHA384}(\text{salt} = \text{nonce},\ ss)$$
$$K = T(1)[0..31], \quad T(1) = \text{HMAC-SHA384}\big(PRK,\ \text{info}\ \|\ \texttt{0x01}\big), \quad \text{info} = \texttt{"BEA-v1|"}\ \|\ \text{mode}\ \|\ \texttt{"|"}\ \|\ \text{challenge\_id}$$

**AES-256-GCM** (NIST SP 800-38D) encrypts and authenticates:

$$(C, T) = \text{AES-GCM}_K\big(IV,\ P,\ AAD\big), \qquad AAD = \text{mode}\ \|\ \texttt{"|"}\ \|\ \text{challenge\_id}$$

with a random 96-bit $IV$ and a 128-bit tag $T$. Any change to $C$, $IV$, $c$, the mode or the
challenge id changes either $K$ or the tag check, and decryption fails.

**Sizes per request:**

| | Classical | PQ |
|---|---|---|
| KEM ciphertext | 32 B (X25519 public key) | 1,088 B |
| Signature | 64 B (Ed25519) | 3,309 B |
| Extra per request in PQ mode | | **+4,301 B** |

**Server key pinning.** At enrolment the client stores the server's KEM and verdict public keys and
refuses any later challenge that names a different KEM key.

**Templates at rest** are encrypted with AES-256-GCM under a server storage key, with
$AAD = \text{user\_id}\ \|\ \texttt{"|"}\ \|\ \text{model\_version}$.

### 3.3 Challenge, liveness actions and single use

**Nonce:** 32 bytes from the OS random generator. **Lifetime:** 60 s (Redis TTL).

**Actions** are chosen by the server from {blink twice, turn left, turn right}, in an order nobody can
predict before the nonce exists:

$$m = \text{HMAC-SHA256}(\text{server\_secret},\ \text{nonce})$$

then a Fisher–Yates shuffle driven by the MAC bytes: for $i = n-1 \dots 1$, swap item $i$ with item
$j = m[n-1-i] \bmod (i+1)$; the first two items are the actions.

**Single use:** the challenge is read and deleted in one atomic Redis `GETDEL`, so two copies of the
same bundle can never both pass; a "used" marker distinguishes REPLAY from EXPIRED.

### 3.4 Face pipeline and detectors (Layer C)

**Face detection — YuNet** (CNN detector, OpenCV `FaceDetectorYN`), score threshold 0.5.
The face used is the **largest** among detections whose score is within 0.1 of the best one,
and it must be at least 112 px wide (this rejects small false boxes without dropping real faces).

**Alignment.** The 5 YuNet landmarks $p_i$ are mapped onto the ArcFace template $q_i$ with a
similarity transform (rotation $R$, scale $s$, translation $t$) fitted robustly (LMedS):

$$\min_{s,R,t} \sum_i \| s R p_i + t - q_i \|^2 \quad\Rightarrow\quad 112\times112 \text{ aligned crop}$$

**Matcher — ArcFace** (ResNet-50, InsightFace buffalo_l). Input $x = (I_{RGB} - 127.5)/127.5$.
The 512-d output $e$ is L2-normalised, and the similarity to the enrolled template $t$ is

$$s_{\text{match}} = \cos(e, t) = \frac{e\cdot t}{\|e\|\,\|t\|}, \qquad \text{accept if } s_{\text{match}} \ge \tau = 0.2192 \ (\text{FMR } 0.1\%)$$

**M1 presentation attacks — MiniFASNet** (Silent-Face-Anti-Spoofing). Two CNNs see the face box
enlarged by 2.7× and 4.0×, resized to 80×80 (raw BGR, 0–255). Their softmax outputs over 3 classes
(class 1 = real) are averaged over models and frames, $\bar p = (p_0, p_1, p_2)$. With the
pretrained models the decision is Silent-Face's own rule ("real" only if class 1 is the top class),
encoded as a score with threshold 0.5:

$$s_{\text{M1}} = 0.5 + \tfrac12\big(\max(p_0, p_2) - p_1\big), \qquad \text{reject if } s_{\text{M1}} > 0.5$$

**M2 morphing — differential MAD (D-MAD) with an RBF SVM.** For enrolled reference $r$ and live
probe $p$ (both ArcFace embeddings):

$$f = e(r) - e(p) \in \mathbb{R}^{512}, \qquad \tilde f = (f - \mu)/\sigma \ \ (\text{standardised})$$
$$d(\tilde f) = \sum_i \alpha_i y_i\, K(\tilde f_i, \tilde f) + b, \qquad K(a, b) = \exp(-\gamma \|a - b\|^2), \quad \gamma = \frac{1}{512\cdot \text{Var}(\tilde f)}$$
$$s_{\text{M2}} = \frac{1}{1 + e^{-d}}, \qquad \text{reject if } s_{\text{M2}} > 0.5608 \ (\text{BPCER } 0.5\%)$$

Class weights are balanced ($w_c = N / (2 N_c)$). A genuine pair gives a small, typical difference
vector; a morphed reference only half-matches the live person and gives a different one.

**M3 deepfakes — GenD** (CLIP ViT-L/14 vision encoder + normalised linear head). Each 224×224 face
crop is normalised with CLIP's mean $\mu$ and std $\sigma$, $x = (I/255 - \mu)/\sigma$; the pooled
feature $h$ is L2-normalised and classified:

$$P(\text{fake}) = \text{softmax}\big(W \tfrac{h}{\|h\|} + b\big)_{\text{fake}}, \qquad s_{\text{M3}} = \text{mean over 3 frames}, \qquad \text{reject if } s_{\text{M3}} > 0.9182$$

**LayerNorm-only tuning** (the deployed M3): only the scale $\gamma$ and shift $\beta$ of every
LayerNorm in the encoder are trained (102,400 of 304 million parameters),

$$\text{LN}(x) = \gamma \odot \frac{x - \mu_x}{\sqrt{\sigma_x^2 + \epsilon}} + \beta$$

with class-weighted cross-entropy $L = -\sum_c w_c\, y_c \log P(c)$, AdamW, fp16, 2 epochs on 80%
of OpenForensics Val (split by source image); the threshold comes from the other 20%.

**M4 replay / liveness — MediaPipe Face Landmarker.** Per frame (15 fps):
blink level $b = \tfrac12(\text{eyeBlinkLeft} + \text{eyeBlinkRight})$ and head yaw from the face
rotation matrix $R$:

$$\text{yaw} = \frac{180}{\pi}\,\operatorname{atan2}(R_{02},\ R_{22})$$

A **blink** is counted when $b$ rises above 0.5 and then falls below 0.3 (hysteresis). A **turn** is
counted when $|\text{yaw}| > 15^\circ$ and reset when $|\text{yaw}| < 7.5^\circ$. The two requested
actions must be seen **in order** (blink twice = two blinks), otherwise $s_{\text{M4}} = 1$ (reject).

**M5 injection (virtual cameras) — rules + gradient-boosted trees.** Rule: a camera name matching a
known virtual camera (OBS Virtual Camera, v4l2loopback, ManyCam, …) scores 1.0. Otherwise a
HistGradientBoosting classifier scores 16 features of the capture, including the inter-frame
intervals $\Delta t_i = t_{i+1} - t_i$:

$$\overline{\Delta t},\ \ \sigma_{\Delta t},\ \ CV = \sigma_{\Delta t}/\overline{\Delta t},\ \ P_5,\ P_{95},\ \ \Pr\big(|\Delta t_i - \text{median}| < 1\,\text{ms}\big)$$

plus the duplicate-frame ratio (identical BLAKE2b hashes of consecutive JPEG frames), requested vs
delivered resolution and frame rate, and the latency when the resolution is changed. Virtual
cameras tend to be too regular, repeat frames, ignore resolution changes and switch instantly.
(Until enough recordings exist, only the name rule is active.)

### 3.5 Decision rule

$$\text{reject if any } s_k > \tau_k \text{ for } k \in \{\text{M5, M3, M1, M2, M4}\}\ \text{(reason = first in this order)};\ \text{otherwise accept iff } s_{\text{match}} \ge \tau_{\text{match}}$$

### 3.6 Signed verdict (the project's novelty)

The server builds a record

`{challenge_id, mode, tx_hash, sample_hash, scores{match, m1..m5}, model_versions, thresholds_version, decision, reason_code, timestamp, signer}`

and signs it with its own key (ML-DSA-65 in PQ mode):

$$\sigma_{\text{verdict}} = \text{Sign}_{sk_{\text{server}}}\big(\text{SHA-384}(\text{canonical\_json}(\text{record}))\big)$$

Anyone with the server's public key can later re-check that the AI's scores and the decision were
not changed, and that they belong to this exact challenge, transaction and video. Only hashes are
stored, never face images. One verdict is signed per consumed challenge, including rejections.

### 3.7 Evaluation algorithms

| Metric | Formula |
|---|---|
| APCER (attacks wrongly accepted) | $\frac{\#\{\text{attacks with } s \le \tau\}}{\#\text{attacks}}$, per attack type |
| BPCER (genuine wrongly rejected) | $\frac{\#\{\text{bona fide with } s > \tau\}}{\#\text{bona fide}}$ |
| EER | the error where APCER = BPCER |
| AUC | $\Pr(s_{\text{attack}} > s_{\text{bona fide}})$, computed from ranks (Mann–Whitney), ties count ½ |
| FMR / FNMR (matcher) | impostor pairs with $s \ge \tau$ / genuine pairs with $s < \tau$ |
| Threshold at BPCER $\beta$ | sort the $n$ bona fide calibration scores ascending, $s_{(1)} \le \dots \le s_{(n)}$; $\tau = s_{(n - \lfloor \beta n \rfloor)}$, so at most $\lfloor \beta n \rfloor$ genuine samples score above it |
| Threshold at FMR $\alpha$ | sort the $n$ impostor similarities; $\tau$ = just above $s_{(n - \lfloor \alpha n \rfloor)}$, so at most $\lfloor \alpha n \rfloor$ impostors reach it |

Protocols: identity-grouped splits (no person in both training and test), 5-fold **GroupKFold** for
M2 with thresholds from an inner cross-validation on the training subjects only, and every module
reported at BPCER 0.5%, 1% and 5%.

---

## 4. Server checks (fixed order, stop at the first failure)

| # | Check | Reason code |
|---|---|---|
| 1 | Challenge exists, not expired, not used → consumed atomically (`GETDEL`) | EXPIRED / REPLAY |
| 2 | Bundle mode = mode stored with the challenge | MODE_MISMATCH |
| 3 | Decapsulate, derive $K$, AES-GCM tag valid | TAMPER |
| 4 | Manifest nonce, tx_hash, challenge_id, mode = issued values | TX_MISMATCH |
| 5 | $\text{SHA-384}(\text{video})$ = manifest.sample_hash | HASH_MISMATCH |
| 6 | Device signature valid under the enrolled device key | SIG_INVALID |

Then: matcher (NO_MATCH), M1 (PAD_FAIL), M2 (MORPH_FLAG), M3 (DEEPFAKE_FAIL), M4 (SCRIPT_FAIL),
M5 (INJECTION_FAIL). The client only sees accepted / rejected; reason codes stay in the server log.

---

## 5. How each attack is countered

| Attack | What the attacker does | Countered by | Algorithm / mechanism |
|---|---|---|---|
| Eavesdropping, "harvest now, decrypt later" | Records the traffic to decrypt it with a future quantum computer | Layer B | ML-KEM-768 key agreement + HKDF-SHA384 + AES-256-GCM; face data never travels in the clear |
| Tampering in transit | Changes bytes of the encrypted package | Check 3 (and 5, 6) | AES-GCM authentication tag; SHA-384 video hash; ML-DSA signature over the manifest |
| Replay of a captured package | Sends a valid package a second time | Check 1 | 32-byte single-use nonce, atomic `GETDEL`, 60 s lifetime |
| Delayed use | Holds a package and sends it later | Check 1 | Redis TTL + expiry timestamp → EXPIRED |
| Re-targeting to another payment | Re-uses evidence to approve a different transaction | Check 4 | tx_hash and nonce inside the signed manifest; challenge_id in HKDF info and GCM AAD |
| Downgrade to classical crypto | Relabels a PQ challenge as CLASSICAL | Check 2 (+3) | Mode stored server-side; mode is part of the HKDF info and the GCM AAD |
| Man-in-the-middle key substitution | Gives the client its own "server" key | Client | Server KEM key pinned at enrolment; challenge refused if it differs |
| Forged or impersonated device | Builds a package without the user's device key | Check 6 | ML-DSA-65 / Ed25519 signature with the enrolled device key |
| Wrong person | Someone else at the camera | Matcher | ArcFace cosine similarity vs the enrolled template, threshold at FMR 0.1% |
| Photo, screen or mask (presentation attack) | Holds a picture or plays a video to the camera | M1, M4 | MiniFASNet multi-scale spoof classifier; random actions a still image cannot perform |
| Pre-recorded video | Plays a recording of the user | M4 | Actions derived by HMAC from a fresh nonce: a recording cannot know them in advance |
| Morphed enrolment photo | Enrols a face blended from two people | M2 | D-MAD: ArcFace difference vector + RBF SVM |
| Deepfake | AI-generated or face-swapped video | M3 | GenD (CLIP ViT-L/14) fake classifier, LayerNorm-tuned |
| Injection through a virtual camera | Feeds a fake stream via OBS / v4l2loopback | M5 | Virtual-camera name rule + gradient-boosted trees on frame-timing and camera-behaviour features |
| Editing a verdict afterwards (insider, repudiation) | Changes a stored score or decision | Signed verdict | ML-DSA-65 signature over SHA-384 of the canonical record; any edit fails verification |
| Stealing stored templates | Copies the database | Server storage | Templates encrypted with AES-256-GCM (AAD = user + model version) |
| Probing the detectors | Learns which check fired from the reply | Server | Generic accepted / rejected reply; reason codes only in the server log |

**Out of scope** (documented limits): a compromised server, side channels, denial of service,
enrolment fraud, a strong attacker on a rooted device, and high-quality 3D masks.

---

## 6. Where to find each piece

| Piece | File |
|---|---|
| Crypto suite (both modes), manifest | `crypto/suite.py`, `crypto/manifest.py` |
| Challenges, actions, single use | `server/challenge.py` |
| Checks 1–6 | `server/checks.py` |
| Decision rule, signed verdict | `server/decision.py`, `server/verdict.py` |
| Face detection, alignment, crops | `preprocessing/faces.py` |
| Matcher, M1–M5 | `detectors/matcher/`, `detectors/m1_pad/`, `detectors/m2_morph/`, `detectors/m3_deepfake/`, `detectors/m4_replay/`, `detectors/m5_injection/` |
| Metrics | `eval/metrics.py` |
| Results (CSV) | `eval/results/`, summarised in `docs/PROGRESS_REVIEW.md` |
