# Runbook: from a fresh clone to every result

Run from the repo root with the virtualenv active. Every number in the report must come
from a CSV in `eval/results/` produced by one of these commands. Benchmark on Linux, WSL2 or Docker.

## 0. Setup

```bash
pip install -r requirements.txt
pytest -q
docker run -d -p 6379:6379 redis:7          # or add --fakeredis to every "python -m server"
```

## 1. Models (S3, S5, S10)

```bash
python scripts/fetch_models.py yunet landmarker buffalo_l
pip install -r requirements-export.txt
python scripts/fetch_models.py minifasnet gend
python scripts/export_onnx.py m1             # -> eval/results/onnx_export.csv (cosine > 0.999, CPU ms)
python scripts/export_onnx.py m3             # also confirms GenD's fake class index
```

## 2. Datasets (S4)

```bash
python scripts/download_frll.py
python scripts/download_frll_morphs.py       # -> data/splits/m2_pairs.csv
python scripts/download_openforensics.py --splits Val Test-Dev   # -> data/splits/m3_val.csv, m3_test-dev.csv
python scripts/download_celeba_spoof.py      # manual Google Drive download; then:
python scripts/filter_celeba_spoof.py        # -> data/splits/celeba_spoof_{train,val,test}.csv
```

## 3. Matcher and detectors (S3, S5, S7)

```bash
python scripts/eval_matcher.py                                                   # matcher.csv, matcher_scores.csv
python scripts/eval_detectors.py m1_pad --csv data/splits/celeba_spoof_val.csv data/splits/celeba_spoof_test.csv --tag zero_shot
python scripts/eval_detectors.py m3_deepfake --csv data/splits/m3_val.csv data/splits/m3_test-dev.csv --precropped --tag zero_shot
python scripts/train_m2.py                                                       # m2_morph.csv, scores_m2_morph.csv
python scripts/train_m1.py && python scripts/export_onnx.py m1 --checkpoints models/weights/m1_pad/finetuned
python scripts/eval_detectors.py m1_pad --csv data/splits/celeba_spoof_val.csv data/splits/celeba_spoof_test.csv --tag finetuned
```

`scores_<module>.csv` is overwritten by the last `eval_detectors.py` run for that module;
run the configuration you will deploy last, before calibrating.

## 4. M5 recordings (S8)

Add each consenting team member to `docs/consent.txt` (one name per line), then for each person:

```bash
python scripts/record_m5.py --label bona_fide --camera 0 --subject <name> --sessions 50
python scripts/record_m5.py --label obs --camera <OBS index> --subject <name> --sessions 50     # OBS Virtual Camera playing a video
python scripts/record_m5.py --label v4l2loopback --camera <N> --subject <name> --sessions 50    # Linux only
python scripts/train_m5.py                                                                    # leave-one-tool-out -> m5_injection.csv
```

## 5. Calibrate and integrate (S11)

```bash
python scripts/calibrate.py                  # thresholds at BPCER 0.5% / FMR 0.1% -> config.yaml, thresholds.csv
python -m server                             # /health must list no "stub" versions
python -m client.enroll --user <name>
python -m client.capture --user <name> --tx '{"type":"payment","amount":500,"currency":"INR","payee":"acct-42"}' --save var/sessions/<name>1
python scripts/verify_verdict.py --all
```

## 6. Replay suite (S9)

```bash
python scripts/replay_attacks.py --runs 1000 --session var/sessions/<name>1     # -> replay.csv
```

Pass condition: every attack row has `correct_code_rate` 1.0 and the genuine rows have
`protocol_false_rejections` 0.

## 7. Evaluation (S12)

```bash
python -m eval.crypto_bench --runs 1000
python -m server &                            # note its PID
python -m eval.perf --runs 500 --session var/sessions/<name>1 --url http://127.0.0.1:8000 --server-pid <PID>
PQBIO_MODE=PQ locust -f eval/locustfile.py --host http://127.0.0.1:8000 --headless -u 8 -r 2 -t 2m --csv eval/results/locust_pq
PQBIO_MODE=CLASSICAL locust -f eval/locustfile.py --host http://127.0.0.1:8000 --headless -u 8 -r 2 -t 2m --csv eval/results/locust_classical
python -m eval.attack_matrix --attack print=<dir> --attack screen=<dir> --attack deepfake=<dir> --attack obs=var/m5_sessions/obs \
    --bona-fide genuine=var/sessions --morph-pairs <csv with morph,probe> --limit 200
python -m eval.plots
```

## 8. Demo (S13)

enrol → accepted transaction → printed photo rejected (PAD_FAIL) → pre-recorded video rejected
(`client.capture --from-session`, SCRIPT_FAIL) → OBS Virtual Camera rejected (INJECTION_FAIL) →
replayed bundle rejected (REPLAY, from `replay_attacks.py`) → `verify_verdict.py --all`.
