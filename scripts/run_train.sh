#!/bin/bash
# scripts/run_train.sh
# Train a DA4JIE model using the specified config.
#
# Usage:
#   MODEL=agie CONFIG=configs/example_agie.json bash scripts/run_train.sh

set -e

CONFIG="${CONFIG:-configs/example_oneie.json}"
PYTHON="${PYTHON:-python}"

export PYTHONPATH=$(pwd):$PYTHONPATH

echo "Training with config: $CONFIG"
"$PYTHON" train.py -c "$CONFIG"
