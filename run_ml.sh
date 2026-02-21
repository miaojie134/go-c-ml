#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

BINANCE_DIR="${BINANCE_DIR:-./binance-public-data}"
BINANCE_PY_DIR="${BINANCE_PY_DIR:-$BINANCE_DIR/python}"
VENV_DIR="${VENV_DIR:-$BINANCE_PY_DIR/.venv}"
VENV_PY="${VENV_PY:-$VENV_DIR/bin/python}"
DATA_ROOT="${DATA_ROOT:-$BINANCE_PY_DIR/data}"
INTERVAL="${INTERVAL:-15m}"
TRADING_TYPE="${TRADING_TYPE:-um}"
TIME_PERIOD="${TIME_PERIOD:-daily}"
START_DATE="${START_DATE:-2023-01-01}"
END_DATE="${END_DATE:-}"

ensure_binance_repo() {
  if [[ -d "$BINANCE_PY_DIR" ]]; then
    return
  fi

  if ! command -v git >/dev/null 2>&1; then
    echo "git not found, cannot auto-download binance-public-data"
    exit 1
  fi

  echo "[setup] cloning binance-public-data ..."
  git clone --depth 1 https://github.com/binance/binance-public-data.git "$BINANCE_DIR"
}

ensure_python() {
  if [[ -x "$VENV_PY" ]]; then
    return
  fi

  local host_py
  host_py="$(command -v python3 || true)"
  if [[ -z "$host_py" ]]; then
    host_py="$(command -v python || true)"
  fi
  if [[ -z "$host_py" ]]; then
    echo "python3/python not found, cannot create venv"
    exit 1
  fi

  echo "[setup] creating venv: $VENV_DIR"
  "$host_py" -m venv "$VENV_DIR"
}

ensure_deps() {
  if "$VENV_PY" - <<'PY' >/dev/null 2>&1
import lightgbm, pandas, numpy, sklearn, pandas_ta
PY
  then
    return
  fi

  echo "[setup] installing dependencies ..."
  "$VENV_PY" -m pip install -U pip
  "$VENV_PY" -m pip install -r requirements.txt
  if [[ -f "$BINANCE_PY_DIR/requirements.txt" ]]; then
    "$VENV_PY" -m pip install -r "$BINANCE_PY_DIR/requirements.txt"
  fi
}

ensure_runtime() {
  ensure_binance_repo
  ensure_python
  ensure_deps
}

require_symbol() {
  local command_name="$1"
  shift
  if [[ $# -eq 0 || "$1" == -* ]]; then
    echo "error: SYMBOL is required for '$command_name'" >&2
    echo "example: ./run_ml.sh $command_name ETHUSDT" >&2
    return 1
  fi
  REQUIRED_SYMBOL="$1"
  return 0
}

resolve_common_overrides() {
  local interval="$INTERVAL"
  local trading_type="$TRADING_TYPE"
  local time_period="$TIME_PERIOD"
  local start_date="$START_DATE"
  local end_date="$END_DATE"

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --interval)
        interval="${2:-$interval}"
        shift 2
        ;;
      --trading-type)
        trading_type="${2:-$trading_type}"
        shift 2
        ;;
      --time-period)
        time_period="${2:-$time_period}"
        shift 2
        ;;
      --start-date)
        start_date="${2:-$start_date}"
        shift 2
        ;;
      --end-date)
        end_date="${2:-$end_date}"
        shift 2
        ;;
      *)
        shift
        ;;
    esac
  done

  echo "$interval|$trading_type|$time_period|$start_date|$end_date"
}

has_kline_data() {
  local symbol_upper="$1"
  local interval="$2"
  local trading_type="$3"
  local time_period="$4"
  local pattern="${DATA_ROOT}/futures/${trading_type}/${time_period}/klines/${symbol_upper}/${interval}/${symbol_upper}-${interval}-*.zip"

  shopt -s nullglob
  local files=( $pattern )
  shopt -u nullglob
  [[ ${#files[@]} -gt 0 ]]
}

ensure_kline_data() {
  local symbol="$1"
  local interval="$2"
  local trading_type="$3"
  local time_period="$4"
  local start_date="$5"
  local end_date="$6"

  local symbol_upper
  symbol_upper="$(echo "$symbol" | tr '[:lower:]' '[:upper:]')"

  if has_kline_data "$symbol_upper" "$interval" "$trading_type" "$time_period"; then
    return
  fi

  if [[ ! -f "$BINANCE_PY_DIR/download-kline.py" ]]; then
    echo "download script not found: $BINANCE_PY_DIR/download-kline.py"
    exit 1
  fi

  if [[ "$(basename "$DATA_ROOT")" != "data" ]]; then
    echo "DATA_ROOT must end with '/data' for auto download, current: $DATA_ROOT"
    exit 1
  fi

  local store_dir
  store_dir="$(dirname "$DATA_ROOT")"
  local dl_cmd=( "$VENV_PY" "$BINANCE_PY_DIR/download-kline.py" -t "$trading_type" -s "$symbol_upper" -i "$interval" -startDate "$start_date" )
  if [[ -n "$end_date" ]]; then
    dl_cmd+=( -endDate "$end_date" )
  fi
  if [[ "$time_period" == "daily" ]]; then
    dl_cmd+=( -skip-monthly 1 )
  elif [[ "$time_period" == "monthly" ]]; then
    dl_cmd+=( -skip-daily 1 )
  fi

  echo "[setup] data not found, downloading klines: symbol=$symbol_upper interval=$interval type=$trading_type period=$time_period"
  STORE_DIRECTORY="$store_dir" "${dl_cmd[@]}"

  if ! has_kline_data "$symbol_upper" "$interval" "$trading_type" "$time_period"; then
    echo "download finished but still no data found for $symbol_upper ($trading_type/$time_period/$interval)"
    exit 1
  fi
}

to_prefix() {
  local symbol="$1"
  local base="${symbol%USDT}"
  if [[ -z "$base" || "$base" == "$symbol" ]]; then
    base="$symbol"
  fi
  echo "$base" | tr '[:upper:]' '[:lower:]'
}

usage() {
  cat <<'EOF'
Usage:
  ./run_ml.sh <command> SYMBOL [extra args...]

Commands:
  train         Train one model
  auto          Auto-search long_only config
  auto-ls       Auto-search long_short config
  backtest      Backtest quant model
  backtest-best Backtest best long_short model
  grid          Grid-search thresholds on quant model

Examples:
  ./run_ml.sh train ETHUSDT
  ./run_ml.sh train BTCUSDT --start-date 2022-01-01
  ./run_ml.sh auto-ls ETHUSDT
  ./run_ml.sh backtest ETHUSDT
EOF
}

if [[ $# -eq 0 || "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

cmd="$1"
shift

case "$cmd" in
  train)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    echo "[train] symbol=$symbol interval=$eff_interval start_date=$eff_start_date device=cuda out_prefix=$prefix"
    train_cmd=( "$VENV_PY" scripts/train_lightgbm.py \
      --data-root "$DATA_ROOT" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
      --model-out "./models/${prefix}_quant_model.txt" \
      --metrics-out "./models/${prefix}_metrics.json" \
      --importance-out "./models/${prefix}_feature_importance.csv" )
    if [[ -n "$eff_end_date" ]]; then
      train_cmd+=( --end-date "$eff_end_date" )
    fi
    train_cmd+=( "$@" )
    "${train_cmd[@]}"
    ;;

  auto)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    echo "[auto] symbol=$symbol mode=long_only interval=$eff_interval start_date=$eff_start_date device=cuda"
    auto_cmd=( "$VENV_PY" scripts/auto_search_eth.py \
      --data-root "$DATA_ROOT" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
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
      --results-out "./models/${prefix}_auto_search_results.csv" \
      --best-json-out "./models/${prefix}_best_config.json" \
      --best-model-out "./models/${prefix}_best_model.txt" \
      --best-equity-out "./models/${prefix}_best_equity.csv" )
    if [[ -n "$eff_end_date" ]]; then
      auto_cmd+=( --end-date "$eff_end_date" )
    fi
    auto_cmd+=( "$@" )
    "${auto_cmd[@]}"
    ;;

  auto-ls)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    echo "[auto] symbol=$symbol mode=long_short interval=$eff_interval start_date=$eff_start_date device=cuda"
    auto_ls_cmd=( "$VENV_PY" scripts/auto_search_eth.py \
      --data-root "$DATA_ROOT" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
      --train-ratio 0.7 \
      --val-ratio 0.85 \
      --horizons 1,2,3,4,6 \
      --target-thresholds 0.0,0.0005,0.001 \
      --learning-rates 0.02,0.05 \
      --num-leaves 31,63,127 \
      --n-estimators 700 \
      --early-stop-rounds 80 \
      --buy-min 0.52 \
      --buy-max 0.80 \
      --buy-step 0.02 \
      --sell-min 0.20 \
      --sell-max 0.48 \
      --sell-step 0.02 \
      --fee-bps 5 \
      --slippage-bps 1 \
      --min-trades 20 \
      --optimize strategy_total_return \
      --position-mode long_short \
      --results-out "./models/${prefix}_auto_search_ls_results.csv" \
      --best-json-out "./models/${prefix}_best_ls_config.json" \
      --best-model-out "./models/${prefix}_best_ls_model.txt" \
      --best-equity-out "./models/${prefix}_best_ls_equity.csv" )
    if [[ -n "$eff_end_date" ]]; then
      auto_ls_cmd+=( --end-date "$eff_end_date" )
    fi
    auto_ls_cmd+=( "$@" )
    "${auto_ls_cmd[@]}"
    ;;

  backtest)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    echo "[backtest] symbol=$symbol model=./models/${prefix}_quant_model.txt"
    backtest_cmd=( "$VENV_PY" scripts/backtest_lightgbm.py \
      --data-root "$DATA_ROOT" \
      --model-path "./models/${prefix}_quant_model.txt" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
      --buy-threshold 0.55 \
      --sell-threshold 0.45 \
      --fee-bps 5 \
      --slippage-bps 1 \
      --summary-out "./models/${prefix}_backtest_summary.json" \
      --equity-out "./models/${prefix}_backtest_equity.csv" )
    if [[ -n "$eff_end_date" ]]; then
      backtest_cmd+=( --end-date "$eff_end_date" )
    fi
    backtest_cmd+=( "$@" )
    "${backtest_cmd[@]}"
    ;;

  backtest-best)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    best_model="./models/${prefix}_best_ls_refine_model.txt"
    if [[ ! -f "$best_model" ]]; then
      best_model="./models/${prefix}_best_ls_model.txt"
    fi
    if [[ ! -f "$best_model" ]]; then
      echo "best model not found: ./models/${prefix}_best_ls_refine_model.txt or ./models/${prefix}_best_ls_model.txt"
      echo "run auto-ls first: ./run_ml.sh auto-ls $symbol"
      exit 1
    fi
    echo "[backtest] symbol=$symbol model=$best_model mode=long_short"
    backtest_best_cmd=( "$VENV_PY" scripts/backtest_lightgbm.py \
      --data-root "$DATA_ROOT" \
      --model-path "$best_model" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
      --position-mode long_short \
      --buy-threshold 0.54 \
      --sell-threshold 0.14 \
      --fee-bps 5 \
      --slippage-bps 1 \
      --summary-out "./models/${prefix}_best_live_summary.json" \
      --equity-out "./models/${prefix}_best_live_equity.csv" )
    if [[ -n "$eff_end_date" ]]; then
      backtest_best_cmd+=( --end-date "$eff_end_date" )
    fi
    backtest_best_cmd+=( "$@" )
    "${backtest_best_cmd[@]}"
    ;;

  grid)
    require_symbol "$cmd" "$@" || exit 2
    symbol="$REQUIRED_SYMBOL"
    shift
    ensure_runtime
    common="$(resolve_common_overrides "$@")"
    IFS='|' read -r eff_interval eff_trading_type eff_time_period eff_start_date eff_end_date <<< "$common"
    ensure_kline_data "$symbol" "$eff_interval" "$eff_trading_type" "$eff_time_period" "$eff_start_date" "$eff_end_date"
    prefix="$(to_prefix "$symbol")"
    echo "[grid] symbol=$symbol model=./models/${prefix}_quant_model.txt"
    grid_cmd=( "$VENV_PY" scripts/grid_search_backtest.py \
      --data-root "$DATA_ROOT" \
      --model-path "./models/${prefix}_quant_model.txt" \
      --symbol "$symbol" \
      --interval "$eff_interval" \
      --trading-type "$eff_trading_type" \
      --time-period "$eff_time_period" \
      --start-date "$eff_start_date" \
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
      --results-out "./models/${prefix}_grid_results.csv" \
      --best-out "./models/${prefix}_grid_best.json" \
      --best-equity-out "./models/${prefix}_grid_best_equity.csv" )
    if [[ -n "$eff_end_date" ]]; then
      grid_cmd+=( --end-date "$eff_end_date" )
    fi
    grid_cmd+=( "$@" )
    "${grid_cmd[@]}"
    ;;

  *)
    echo "unknown command: $cmd"
    echo
    usage
    exit 1
    ;;
esac
