#!/usr/bin/env bash
# Bring MFCC-CNN and the Transformer onto the v2 dataset so all three models are
# compared on the same 3220-window test split, then export one bundle per model for
# the multi-model server. The v2 CRNN itself is already trained (run_v2_pipeline.sh).
#
# Usage:  bash scripts/run_v2_all_models.sh
# Log:    tee'd to reports/v2_all_models.log

set -euo pipefail
cd "$(dirname "$0")/.."

export SAFESCAPE_PROC_DIR=data/processed_v2
PY=../.venv/Scripts/python.exe
CK=../models/checkpoints
EX=../models/exported
LOG=reports/v2_all_models.log

exec > >(tee -a "$LOG") 2>&1

step() { echo ""; echo "=== [$(date +%H:%M:%S)] $* ==="; }

cd scripts

step "1/5 train MFCC-CNN on v2 (v1 Optuna winner)"
$PY 06_train.py --arch mfcc_cnn --epochs 25 --lr 1.25e-3 --batch-size 64 --dropout 0.253 \
    --ckpt $CK/mfcc_cnn_v2.pt --log-csv ../reports/mfcc_cnn_v2_train_log.csv

step "2/5 distil Transformer from the v2 CRNN"
$PY 08_train_distilled_transformer.py --teacher $CK/logmel_crnn_v2.pt \
    --ckpt $CK/transformer_v2.pt --log-csv ../reports/transformer_v2_train_log.csv

step "3/5 safety-constrained calibration per model"
mkdir -p $EX/mfcc_cnn $EX/logmel_crnn $EX/transformer
$PY 12_calibrate_logits.py --arch mfcc_cnn --ckpt $CK/mfcc_cnn_v2.pt --out $EX/mfcc_cnn/calibration.json
$PY 12_calibrate_logits.py --arch transformer --ckpt $CK/transformer_v2.pt --out $EX/transformer/calibration.json
cp $EX/calibration_v2.json $EX/logmel_crnn/calibration.json

step "4/5 evaluate all three on the v2 test split"
$PY 10_evaluate.py --arch mfcc_cnn --ckpt $CK/mfcc_cnn_v2.pt --tag mfcc_cnn_v2 --cpu-only \
    --calibration $EX/mfcc_cnn/calibration.json --metrics-json ../reports/metrics/mfcc_cnn_v2.json
$PY 10_evaluate.py --arch logmel_crnn --hidden-size 128 --ckpt $CK/logmel_crnn_v2.pt \
    --tag logmel_crnn_v2_calibrated --cpu-only --calibration $EX/logmel_crnn/calibration.json \
    --metrics-json ../reports/metrics/logmel_crnn_v2.json
$PY 10_evaluate.py --arch transformer --ckpt $CK/transformer_v2.pt --tag transformer_v2 --cpu-only \
    --calibration $EX/transformer/calibration.json --metrics-json ../reports/metrics/transformer_v2.json

step "4b/5 re-evaluate on the leak-free test split (headline numbers for dashboard/report)"
# 230 distress_call test windows duplicate train/val audio; see 18_flag_duplicate_sources.py
$PY 18_flag_duplicate_sources.py
DEDUP=../data/processed_v2/windows_manifest_dedup.csv
for m in mfcc_cnn logmel_crnn transformer; do
    mv ../reports/metrics/${m}_v2.json ../reports/metrics/${m}_v2_full.json
done
$PY 10_evaluate.py --arch mfcc_cnn --ckpt $CK/mfcc_cnn_v2.pt --tag mfcc_cnn_v2_leakfree --cpu-only \
    --calibration $EX/mfcc_cnn/calibration.json --manifest $DEDUP --metrics-json ../reports/metrics/mfcc_cnn_v2.json
# the CRNN is served int8, so evaluate the served bundle rather than the fp32 checkpoint
$PY 11_quantize_export.py quantize --arch logmel_crnn --hidden-size 128 --ckpt $CK/logmel_crnn_v2.pt \
    --out ../models/quantized/logmel_crnn_v2_quant.pt
$PY 11_quantize_export.py export --arch logmel_crnn --hidden-size 128 --quantized \
    --quant-path ../models/quantized/logmel_crnn_v2_quant.pt --out-dir $EX/logmel_crnn
$PY 10_evaluate.py --arch logmel_crnn --hidden-size 128 --ckpt $EX/logmel_crnn/best_model.pt --quantized \
    --tag logmel_crnn_v2_int8_leakfree --calibration $EX/logmel_crnn/calibration.json \
    --manifest $DEDUP --metrics-json ../reports/metrics/logmel_crnn_v2.json
$PY 10_evaluate.py --arch transformer --ckpt $CK/transformer_v2.pt --tag transformer_v2_leakfree --cpu-only \
    --calibration $EX/transformer/calibration.json --manifest $DEDUP --metrics-json ../reports/metrics/transformer_v2.json

step "5/5 export one serving bundle per model"
$PY 11_quantize_export.py quantize --arch logmel_crnn --hidden-size 128 --ckpt $CK/logmel_crnn_v2.pt \
    --out ../models/quantized/logmel_crnn_v2_quant.pt
$PY 11_quantize_export.py export --arch logmel_crnn --hidden-size 128 --quantized \
    --quant-path ../models/quantized/logmel_crnn_v2_quant.pt --out-dir $EX/logmel_crnn
$PY 11_quantize_export.py export --arch mfcc_cnn --ckpt $CK/mfcc_cnn_v2.pt --out-dir $EX/mfcc_cnn
$PY 11_quantize_export.py export --arch transformer --ckpt $CK/transformer_v2.pt --out-dir $EX/transformer

step "DONE"
