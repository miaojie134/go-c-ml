#!/usr/bin/env python3
import argparse
import json
import os

import lightgbm as lgb
import pandas as pd

from ml_common import (
    add_features,
    apply_date_filter,
    collect_trade_stats,
    interval_to_minutes,
    load_kline_zip_files,
    simulate_strategy,
    summarize_strategy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest LightGBM model on Binance kline data")
    p.add_argument("--data-root", required=True, help="path to binance data root")
    p.add_argument("--model-path", required=True, help="path to lightgbm txt model")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--trading-type", default="um", choices=["um", "cm"])
    p.add_argument("--time-period", default="daily", choices=["daily", "monthly"])
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument(
        "--test-ratio",
        type=float,
        default=0.8,
        help="use the last (1-test_ratio) data for backtest to mimic out-of-sample",
    )
    p.add_argument("--buy-threshold", type=float, default=0.55)
    p.add_argument("--sell-threshold", type=float, default=0.45)
    p.add_argument("--position-mode", default="long_only", choices=["long_only", "long_short"])
    p.add_argument("--fee-bps", type=float, default=5.0, help="per transaction cost in bps")
    p.add_argument("--slippage-bps", type=float, default=1.0, help="per transaction cost in bps")
    p.add_argument("--summary-out", default="models/backtest_summary.json")
    p.add_argument("--equity-out", default="models/backtest_equity.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sell_threshold >= args.buy_threshold:
        raise ValueError("sell-threshold must be smaller than buy-threshold")
    if not (0 < args.test_ratio < 1):
        raise ValueError("test-ratio must be in (0, 1)")

    print("[0/4] loading model", flush=True)
    booster = lgb.Booster(model_file=args.model_path)
    feature_names = booster.feature_name()

    print("[1/4] loading kline data", flush=True)
    raw = load_kline_zip_files(
        data_root=args.data_root,
        symbol=args.symbol,
        interval=args.interval,
        trading_type=args.trading_type,
        time_period=args.time_period,
    )
    raw = apply_date_filter(raw, args.start_date, args.end_date)

    print("[2/4] building features", flush=True)
    feat = add_features(raw)
    missing = [c for c in feature_names if c not in feat.columns]
    if missing:
        raise RuntimeError(f"missing features from data pipeline: {missing[:10]}")
    feat = feat.dropna(subset=feature_names).reset_index(drop=True)
    if feat.empty:
        raise RuntimeError("no rows after dropna(feature_names)")

    split = int(len(feat) * args.test_ratio)
    bt = feat.iloc[split:].copy().reset_index(drop=True)
    if bt.empty:
        raise RuntimeError("backtest slice is empty")
    print(f"[2/4] rows total={len(feat)}, backtest={len(bt)}, features={len(feature_names)}", flush=True)

    print("[3/4] predicting + simulating", flush=True)
    bt["proba"] = booster.predict(bt[feature_names])
    bt["ret_1"] = bt["close"].pct_change().fillna(0.0)
    cost_rate = (args.fee_bps + args.slippage_bps) / 10000.0
    work = simulate_strategy(
        frame=bt[["open_time", "close", "ret_1", "proba"]],
        buy_th=args.buy_threshold,
        sell_th=args.sell_threshold,
        cost_rate=cost_rate,
        position_mode=args.position_mode,
    )

    bars_per_year = (60 * 24 * 365) / interval_to_minutes(args.interval)
    stats = summarize_strategy(work, bars_per_year)
    trade_stats = collect_trade_stats(work)
    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "start_time": str(work["open_time"].iloc[0]),
        "end_time": str(work["open_time"].iloc[-1]),
        "bars": int(len(work)),
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "position_mode": args.position_mode,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "trades_turnover_count": int(work["turnover"].sum()),
        "exposure_ratio": float(work["position"].mean()),
        **stats,
        **trade_stats,
    }

    print("[4/4] writing outputs", flush=True)
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.equity_out) or ".", exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    work[
        [
            "open_time",
            "close",
            "proba",
            "position",
            "ret_1",
            "cost",
            "strategy_ret",
            "equity",
            "benchmark_equity",
        ]
    ].to_csv(args.equity_out, index=False)

    print("\n=== Backtest Done ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"summary saved: {args.summary_out}")
    print(f"equity saved: {args.equity_out}")


if __name__ == "__main__":
    main()
