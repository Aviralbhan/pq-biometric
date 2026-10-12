# Live demo runbook

Ten steps, one command each. Everything runs on this laptop with no network access needed
(models, datasets and the NIST vectors are already on disk). Thresholds are the
**preliminary** ones in `config.yaml` (calibrated on validation data only; final
calibration is S11).

## Before the demo (5 minutes)

1. Close other heavy programs. Plug in the laptop. Good, even light on your face.
2. Open two terminals in `C:\claude projects\pq-biometric` and activate the venv in both:
   `.venv\Scripts\activate`
3. Optional proof of no runtime downloads: in both terminals set
   `set PQBIO_OFFLINE_GUARD=1` (any connection other than to this laptop then fails loudly).
4. Have a photo of yourself open on your phone (for step d).

Demo state lives in `var/demo/` (server keys, database, your device keys, last session).
Delete that folder to start from a clean slate.

## Steps

| Step | Terminal | Command | What the audience sees |
|---|---|---|---|
| a | 1 | `python scripts/demo.py server` | Server starts on 127.0.0.1:8000 (in-process Redis). Leave it running. |
| a | 2 | `python scripts/demo.py health` | Every module listed as **real** (no stubs), thresholds version `preliminary-val-...` |
| b | 2 | `python scripts/demo.py enrol` | Webcam window; press **SPACE**. Device keys for PQ (ML-DSA-65) and CLASSICAL (Ed25519) are created and the server's keys pinned. |
| c | 2 | `python scripts/demo.py verify PQ` | Countdown, two random actions (blink twice / turn left / turn right), then **ACCEPTED** and the signed verdict with every score. |
| c | 2 | `python scripts/demo.py verify CLASSICAL` | Same, classical crypto: **ACCEPTED**. |
| d | 2 | `python scripts/demo.py verify PQ` while holding the phone photo to the webcam | **REJECTED**. Reason PAD_FAIL (M1 score shown) and/or SCRIPT_FAIL (a photo cannot do the actions). |
| e | 2 | `python scripts/demo.py replay` | Uses the last genuine bundle: unchanged re-submit → **REPLAY**; one bit flipped → **TAMPER**; evidence re-bound to a 50,000 INR transaction → **TX_MISMATCH**. The client only ever sees "rejected"; the reasons come from the server. |
| f | 2 | `python scripts/demo.py morph` | M2 on a genuine FRLL pair (accept) and an AMSL morph of subjects 001+002 (MORPH_FLAG). Illustration only: the deployed M2 model was trained on all FRLL subjects. |
| g | 2 | `python scripts/demo.py deepfake` | M3 P(fake) for 3 real and 3 fake OpenForensics Test-Dev faces (fixed sample). At BPCER 0.5% some fakes pass; that is the honest operating point (see PROGRESS_REVIEW.md). |
| h | 2 | `python scripts/demo.py injection` | Only if OBS Studio is installed and its Virtual Camera started: **INJECTION_FAIL**. Otherwise it prints "skipped". |
| i | 2 | `python scripts/demo.py verdict` | The latest verdict's ML-DSA / Ed25519 signature is **VALID** offline; a copy of the database with one score edited is **INVALID**. This is the project's novelty: the AI verdict is signed evidence. |
| j | 2 | `python scripts/demo.py crypto` | ML-KEM-768 / ML-DSA-65 vs X25519 / Ed25519 median times and the extra bytes PQ adds per request (from `eval/results/crypto_bench.csv`). |

## If something goes wrong

| Symptom | Fix |
|---|---|
| `health` cannot connect | Step a is still loading models (about 20-40 s); try again. |
| `health` shows a STUB | A weight file is missing from `models/weights/`; see docs/RUNBOOK.md section 1. |
| Genuine attempt rejected with NO_MATCH or MORPH_FLAG | Lighting or position differs a lot from enrolment: sit closer (face about a quarter of the frame width), re-run `enrol` in the demo room's light. |
| Genuine attempt rejected with SCRIPT_FAIL | The actions were not seen in order: do each clearly (turns about 20-30 degrees and back) while its line is highlighted. |
| Webcam window does not open | Another program holds the camera (close Teams/Zoom/OBS preview). |
| `replay` says to run verify first | Run `verify PQ` once; it saves the bundle `replay` needs. |
