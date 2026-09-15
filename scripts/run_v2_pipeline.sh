#!/usr/bin/env bash
# End-to-end v2 rebuild with the expanded dataset, in an isolated data tree.
#
# Everything lands under data/processed_v2/ and models/checkpoints/logmel_crnn_v2.pt,
# so the v1 champion, its manifests and its feature cache are never touched. If v2 loses
# on the numbers, delete the v2 artefacts and nothing else has changed.
#
# Usage:  bash scripts/run_v2_pipeline.sh
# Log:    tee'd to reports/v2_pipeline.log

set -euo pipefail
cd "$(dirname "$0")/.."

export SAFESCAPE_PROC_DIR=data/processed_v2
PY=.venv/Scripts/python.exe
CKPT=models/checkpoints/logmel_crnn_v2.pt
LOG=reports/v2_pipeline.log

mkdir -p reports
exec > >(tee -a "$LOG") 2>&1

step() { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "1/6 build manifest"
(cd scripts && ../$PY 02_build_manifest.py)

step "2/6 preprocess + window + split"
(cd scripts && ../$PY 03_preprocess_audio.py)

step "3/6 extract features"
(cd scripts && ../$PY 04_extract_features.py)

step "4/6 train log-mel CRNN (winning Optuna config)"
(cd scripts && ../$PY 06_train.py --arch logmel_crnn --hidden-size 128 --epochs 25 \
    --lr 8.56e-4 --batch-size 16 --dropout 0.269 \
    --ckpt "../$CKPT" --log-csv ../reports/logmel_crnn_v2_train_log.csv)

step "5/6 evaluate v2 (uncalibrated) on held-out test"
(cd scripts && ../$PY 10_evaluate.py --arch logmel_crnn --hidden-size 128 \
    --ckpt "../$CKPT" --tag logmel_crnn_v2 --cpu-only)

step "6/6 fit safety-constrained calibration + evaluate calibrated"
(cd scripts && ../$PY 12_calibrate_logits.py --arch logmel_crnn \
    --ckpt "../$CKPT" --objective safety \
    --out ../models/exported/calibration_v2.json)
(cd scripts && ../$PY 10_evaluate.py --arch logmel_crnn --hidden-size 128 \
    --ckpt "../$CKPT" --tag logmel_crnn_v2_calibrated --cpu-only \
    --calibration ../models/exported/calibration_v2.json)

step "DONE — compare against the v1 baseline"
cat <<'EOF'
v1 champion (calibrated, held-out test):
  accuracy 0.848 | macro-F1 0.642 | macro recall 0.784
  distress_call P/R 0.840 / 0.856
  glass_break   P/R 0.283 / 0.895
  horn_skid     P/R 0.246 / 0.638
  alarm         P/R 0.694 / 0.670

v2 numbers are appended to reports/eval_results.md under the tags
logmel_crnn_v2 and logmel_crnn_v2_calibrated.

Adopt v2 ONLY if the weak-class precision improves without losing distress_call recall.
To adopt: re-run 11_quantize_export.py against the v2 checkpoint, copy
calibration_v2.json over models/exported/calibration.json, refresh the numbers in
scripts/14_build_presentation.py, and re-run it.
To reject: delete data/processed_v2, models/checkpoints/logmel_crnn_v2.pt and
models/exported/calibration_v2.json. Nothing else was modified.
EOF
