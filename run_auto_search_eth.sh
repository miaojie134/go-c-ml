#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

VENV_PY="./binance-public-data/python/.venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "python not found: $VENV_PY"
  exit 1
fi

"$VENV_PY" auto_search_eth.py \
  --data-root ./binance-public-data/python/data \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --train-ratio 0.7 \
  --val-ratio 0.85 \
  --horizons 1,2,3,4 \
  --target-thresholds 0.0,0.0005,0.001 \
  --learning-rates 0.02,0.05 \
  --num-leaves 31,63,127 \
  --n-estimators 700 \
  --early-stop-rounds 80 \
  --buy-min 0.52 \
  --buy-max 0.80 \
  --buy-step 0.02 \
  --sell-min 0.20 \
  --sell-max 0.50 \
  --sell-step 0.02 \
  --fee-bps 5 \
  --slippage-bps 1 \
  --min-trades 8 \
  --optimize strategy_total_return \
  --results-out ./models/eth_auto_search_results.csv \
  --best-json-out ./models/eth_best_config.json \
  --best-model-out ./models/eth_best_model.txt \
  --best-equity-out ./models/eth_best_equity.csv
