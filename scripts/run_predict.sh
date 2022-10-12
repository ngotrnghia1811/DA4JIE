#!/bin/bash
# scripts/run_predict.sh
# Run inference using a trained model.
#
# Usage:
#   MODEL_PATH=outputs/best.role.mdl bash scripts/run_predict.sh

set -e

MODEL_PATH="${MODEL_PATH:-outputs/best.role.mdl}"
INPUT_DIR="${INPUT_DIR:-data/input}"
OUTPUT_DIR="${OUTPUT_DIR:-data/output}"
FORMAT="${FORMAT:-ltf}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH=$(pwd):$PYTHONPATH

mkdir -p "$OUTPUT_DIR"

echo "Running inference ..."
"$PYTHON" predict.py \
    -m "$MODEL_PATH" \
    -i "$INPUT_DIR" \
    -o "$OUTPUT_DIR" \
    --format "$FORMAT" \
    --gpu
