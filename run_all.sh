#!/usr/bin/env bash
set -euo pipefail

DATA="data/processed/tcga_lung.h5"

if [ ! -f "$DATA" ]; then
  echo "missing $DATA"
  exit 1
fi

python -m src.pipeline
