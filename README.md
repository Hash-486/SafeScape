# SafeScape — On-Device Acoustic Distress and Hazard Recognition

Implementation for the NN & DL final-year case study. Classifies short (1s) audio
windows into 5 classes: `distress_call`, `glass_break`, `horn_skid`, `alarm`, `ambience`.

## Results

Three architectures trained and compared on the expanded (v2) dataset with one
source-file-disjoint 70/15/15 split. 230 test windows turned out to be byte-identical
to training audio (the Kaggle scream set ships duplicates), so every number below is on
the remaining **leak-free test set of 2,990 windows** — see
`scripts/18_flag_duplicate_sources.py`. Each model has its own safety-constrained
calibration. Per-class tables and confusion matrices: `reports/metrics/*_v2.json`,
[`reports/eval_results.md`](reports/eval_results.md) and the case study report.

| architecture | owner | accuracy | macro-F1 | distress_call recall | served size | CPU latency / window |
|---|---|---|---|---|---|---|
| MFCC-CNN | Amruth Rohan KR | 79.2% | 0.725 | 80.3% | 249 KB | 0.52 ms |
| **log-mel CRNN (champion)** | Harish Venkat VS | **89.7%** | **0.853** | **86.5%** | **516 KB** (int8) | 6.90 ms |
| Distilled Transformer | Harish Venkat VS | 83.8% | 0.774 | 82.5% | 340 KB | 0.48 ms |

All three are served by the FastAPI app; the quantized CRNN is the default.

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Requires an NVIDIA GPU + CUDA-capable PyTorch build for training (see `requirements.txt`);
falls back to CPU automatically if unavailable. Inference is CPU-only by design (matches
the proposal's low-cost/on-device framing).

## Pipeline (run in order from `scripts/`)

1. `01_download_notes.md` — data provenance / commands used to populate `data/raw/`
2. `02_build_manifest.py` — maps every source file to one of the 5 classes, caps ambience
3. `03_preprocess_audio.py` — resample/mono/denoise/window (1.0s, 50% overlap), stratified
   70/15/15 split done at the SOURCE-FILE level (no window-leakage across splits)
4. `06_train.py --arch {mfcc_cnn,logmel_crnn,transformer}` — trains one architecture,
   checkpointing on validation macro-recall
5. `09_hparam_search.py --arch <arch>` — small Optuna search, then retrains the winning config
6. `10_evaluate.py --arch <arch> --ckpt <path> --tag <name>` — test-set metrics, confusion
   matrix, latency, size; appends to `reports/eval_results.md`
7. `11_quantize_export.py quantize --arch <arch>` then `... export --arch <arch> [--quantized]`
   — dynamic PTQ (qint8) + exports a model bundle (`--out-dir`)

The v2 dataset rebuild is `bash scripts/run_v2_pipeline.sh` (CRNN) followed by
`bash scripts/run_v2_all_models.sh` (MFCC-CNN + Transformer retrain, calibration of all
three, leak-free evaluation, one bundle per model under `models/exported/<arch>/`).
`17_make_ui_testset.py` cuts the 20 held-out clips in `demo_clips/test/`;
`19_build_report.py` and `20_build_review_decks.py` regenerate the report and the
Review 2 / Review 3 decks from the metrics.

## Serving + web app

```
python run_demo.py            # starts uvicorn on 0.0.0.0:8124 and verifies it
```

- `http://127.0.0.1:8124/` — mobile-styled web app (Home / History / Settings)
- `http://127.0.0.1:8124/compare` — comparison dashboard: one clip through all three
  models side by side, a UI test run over the held-out clips, and the offline metrics

`/predict?model=<mfcc_cnn|logmel_crnn|transformer>` classifies with one model (CRNN by
default); `/predict/all` with all three. Tests: `server/test_multi_model.py`,
`server/test_clip_position.py`, and `server/test_client.py` against a running server.

## Data sources

Public datasets only (no self-recording, given the deadline) — see
`scripts/01_download_notes.md` for exact commands and licenses: ESC-50 (glass-break,
siren, car-horn, clock-alarm, plus ambience categories), a Kaggle human-screaming
detection dataset (primary `distress_call` source), and RAVDESS speech (Zenodo,
emotion-coded angry/fearful clips) as a distress-adjacent supplement.

## Known limitations / future work

- Distress-call data leans on acted/emotional-speech sources rather than self-recorded,
  India-context audio as originally proposed — flagged as future work.
- Serving is a local FastAPI server on a laptop (LAN-reachable), not true on-device
  inference on the phone itself.
- Dynamic PTQ mainly benefits Linear/RNN layers; the CNN's conv-heavy body may show a
  smaller size/latency win than the CRNN — static quantization/QAT is a documented next step.
- Distilled transformer architecture is a stretch goal, attempted only after both
  mandatory architectures (MFCC-CNN, log-mel CRNN) are trained and evaluated.

## Docs

`docs/` holds the course paperwork: the proposal, the Review 1 / 2 / 3 decks and the
case study report — Review 2 material in `docs/Review2/`, Review 3 (deck, report,
demo steps) in `docs/Review3/`. `data/` (raw + preprocessed audio, ~3.8GB) is not
tracked in git — regenerate it via `scripts/01_download_notes.md` and the
`02`/`03` pipeline steps above.

## License

MIT — see [`LICENSE`](LICENSE).
