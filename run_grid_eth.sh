#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_PY="./binance-public-data/python/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "python not found: $VENV_PY"
  exit 1
fi

"$VENV_PY" grid_search_backtest.py \
  --data-root ./binance-public-data/python/data \
  --model-path ./models/eth_quant_model.txt \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --test-ratio 0.8 \
  --fee-bps 5 \
  --slippage-bps 1 \
  --buy-min 0.52 \
  --buy-max 0.72 \
  --buy-step 0.02 \
  --sell-min 0.30 \
  --sell-max 0.50 \
  --sell-step 0.02 \
  --min-trades 20 \
  --optimize sharpe \
  --top-k 15 \
  --results-out ./models/eth_grid_results.csv \
  --best-out ./models/eth_grid_best.json \
  --best-equity-out ./models/eth_grid_best_equity.csv
