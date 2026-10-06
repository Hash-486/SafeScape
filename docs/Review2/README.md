# Review 2 — one model per member, with the dataset (5 marks)

| file | what |
|---|---|
| `SafeScape_Review2.pptx` | 12-slide deck: dataset, split + duplicate audit, feature pipeline, one slide per model, training curves, confusion matrices, comparison |

| member | model | checkpoint |
|---|---|---|
| Amruth Rohan KR | MFCC-CNN | `models/checkpoints/mfcc_cnn_v2.pt` |
| Harish Venkat VS | log-mel CRNN | `models/exported/logmel_crnn/best_model.pt` (int8, as served) |
| Harish Venkat VS | Distilled Transformer | `models/checkpoints/transformer_v2.pt` |

## Live demo

Each member evaluates their own model on the leak-free held-out test split (2,990 windows).
From `scripts/` in PowerShell:

```
$env:SAFESCAPE_PROC_DIR = "data/processed_v2"
$D = "../data/processed_v2/windows_manifest_dedup.csv"

# Amruth
../.venv/Scripts/python.exe 10_evaluate.py --arch mfcc_cnn --ckpt ../models/checkpoints/mfcc_cnn_v2.pt --calibration ../models/exported/mfcc_cnn/calibration.json --manifest $D --cpu-only --no-log

# Harish
../.venv/Scripts/python.exe 10_evaluate.py --arch logmel_crnn --hidden-size 128 --ckpt ../models/exported/logmel_crnn/best_model.pt --quantized --calibration ../models/exported/logmel_crnn/calibration.json --manifest $D --no-log
../.venv/Scripts/python.exe 10_evaluate.py --arch transformer --ckpt ../models/checkpoints/transformer_v2.pt --calibration ../models/exported/transformer/calibration.json --manifest $D --cpu-only --no-log
```

Expected: accuracy 0.792 / 0.897 / 0.838. `--no-log` prints the metrics without touching
`reports/`.

Then the same model on single clips: `python run_demo.py` from the repo root, open
`http://127.0.0.1:8124/compare`, pick a held-out test clip — each member's card shows
their model's verdict.
