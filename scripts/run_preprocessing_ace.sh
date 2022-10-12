#!/bin/bash
# scripts/run_preprocessing_ace.sh
# Preprocess ACE 2005-E+ data into the OneIE JSON format.
#
# Prerequisites:
#   - LDC2006T06 data at $ACE_DIR
#   - BERT model downloaded at $BERT_CACHE_DIR
#
# Usage:
#   bash scripts/run_preprocessing_ace.sh

set -e

ACE_DIR="${ACE_DIR:-/path/to/LDC2006T06/data}"
OUTPUT_DIR="${OUTPUT_DIR:-data/ace05-e+}"
BERT_MODEL="${BERT_MODEL:-bert-large-cased}"
BERT_CACHE="${BERT_CACHE:-/path/to/bert_cache}"
SPLIT_DIR="resource/splits/ACE05-E+"

mkdir -p "$OUTPUT_DIR"

echo "Processing ACE 2005-E+ ..."
python preprocessing/process_ace.py \
    -i "$ACE_DIR" \
    -o "$OUTPUT_DIR" \
    -s "$SPLIT_DIR" \
    -b "$BERT_MODEL" \
    -c "$BERT_CACHE" \
    -l english

echo "Done. Output at: $OUTPUT_DIR"
