#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_PY="./binance-public-data/python/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "python not found: $VENV_PY"
  exit 1
fi

"$VENV_PY" train_lightgbm.py \
  --data-root ./binance-public-data/python/data \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --model-out ./models/eth_quant_model.txt \
  --metrics-out ./models/eth_metrics.json \
  --importance-out ./models/eth_feature_importance.csv
