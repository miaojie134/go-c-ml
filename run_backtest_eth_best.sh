#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_PY="./binance-public-data/python/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "python not found: $VENV_PY"
  exit 1
fi

"$VENV_PY" backtest_lightgbm.py \
  --data-root ./binance-public-data/python/data \
  --model-path ./models/eth_best_ls_refine_model.txt \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --position-mode long_short \
  --buy-threshold 0.54 \
  --sell-threshold 0.14 \
  --fee-bps 5 \
  --slippage-bps 1 \
  --summary-out ./models/eth_best_live_summary.json \
  --equity-out ./models/eth_best_live_equity.csv
