# SafeScape — On-Device Acoustic Distress and Hazard Recognition

Implementation for the NN & DL final-year case study. Classifies short (1s) audio
windows into 5 classes: `distress_call`, `glass_break`, `horn_skid`, `alarm`, `ambience`.

## Results

Three architectures trained and compared on an identical, source-file-disjoint
70/15/15 split (test set: 2799 windows). Full per-class breakdown, confusion
matrices, and discussion in [`reports/eval_results.md`](reports/eval_results.md).

| architecture | test accuracy | distress_call recall | size (best) | CPU latency |
|---|---|---|---|---|
| MFCC-CNN | 66.6% | 66.7% | 249.3 KB | 1.05 ms |
| **log-mel CRNN (champion)** | **84.5%** | **82.4%** | **516.4 KB** (quantized, -73.6%) | 89–138 ms |
| Transformer (distilled) | 73.5% | 68.7% | 339.0 KB | 6.74 ms |

The quantized log-mel CRNN is the model exported and served by the FastAPI app.

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
   — dynamic PTQ (qint8) + exports the champion model bundle to `models/exported/`

## Serving + web app

```
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Open `http://<your-lan-ip>:8000` on a phone browser on the same Wi-Fi (or a resized
desktop browser) for the mobile-styled web app (Home / History / Settings).

`server/test_client.py` runs a one-clip-per-class smoke test against a running server.

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

`docs/` holds the original case-study proposal and the review presentation template
(course paperwork, not code). `data/` (raw + preprocessed audio, ~3.8GB) is not
tracked in git — regenerate it via `scripts/01_download_notes.md` and the
`02`/`03` pipeline steps above.

## License

MIT — see [`LICENSE`](LICENSE).
