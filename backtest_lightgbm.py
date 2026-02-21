#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any

import numpy as np
import pandas as pd
import lightgbm as lgb

from train_lightgbm import add_features, load_kline_zip_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Backtest LightGBM model on Binance kline data")
    p.add_argument("--data-root", required=True, help="path to binance-public-data/python/data")
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


def interval_to_minutes(interval: str) -> int:
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 60 * 24
    raise ValueError(f"unsupported interval for annualization: {interval}")


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return float(dd.min())


def make_position(proba: np.ndarray, buy_th: float, sell_th: float) -> np.ndarray:
    pos = np.zeros_like(proba, dtype=np.int8)
    state = 0
    for i, p in enumerate(proba):
        if state == 0 and p >= buy_th:
            state = 1
        elif state == 1 and p <= sell_th:
            state = 0
        pos[i] = state
    return pos


def make_position_long_short(proba: np.ndarray, long_th: float, short_th: float) -> np.ndarray:
    pos = np.zeros_like(proba, dtype=np.int8)
    for i, p in enumerate(proba):
        if p >= long_th:
            pos[i] = 1
        elif p <= short_th:
            pos[i] = -1
        else:
            pos[i] = 0
    return pos


def collect_trade_stats(df: pd.DataFrame) -> dict[str, Any]:
    transitions = df["position"].diff().fillna(df["position"]).astype(int)
    entry_idx = list(df.index[transitions == 1])
    exit_idx = list(df.index[transitions == -1])

    if not entry_idx:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
        }

    if len(exit_idx) < len(entry_idx):
        exit_idx.append(df.index[-1])

    trade_returns = []
    for ent, ex in zip(entry_idx, exit_idx):
        if ex <= ent:
            continue
        seg = df.loc[ent:ex]
        r = float((1.0 + seg["strategy_ret"]).prod() - 1.0)
        trade_returns.append(r)

    if not trade_returns:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
        }

    wins = sum(1 for r in trade_returns if r > 0)
    return {
        "closed_trades": int(len(trade_returns)),
        "win_rate": float(wins / len(trade_returns)),
        "avg_trade_return": float(np.mean(trade_returns)),
        "median_trade_return": float(np.median(trade_returns)),
    }


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
    if args.start_date:
        raw = raw[raw["open_time"] >= pd.Timestamp(args.start_date, tz="UTC")]
    if args.end_date:
        raw = raw[raw["open_time"] <= pd.Timestamp(args.end_date, tz="UTC")]

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
    X = bt[feature_names]
    proba = booster.predict(X)
    bt["proba"] = proba
    if args.position_mode == "long_short":
        bt["position"] = make_position_long_short(proba, args.buy_threshold, args.sell_threshold)
    else:
        bt["position"] = make_position(proba, args.buy_threshold, args.sell_threshold)

    bt["ret_1"] = bt["close"].pct_change().fillna(0.0)
    bt["position_prev"] = bt["position"].shift(1).fillna(0).astype(int)
    bt["turnover"] = (bt["position"] - bt["position_prev"]).abs()
    cost_rate = (args.fee_bps + args.slippage_bps) / 10000.0
    bt["cost"] = bt["turnover"] * cost_rate
    bt["strategy_ret"] = bt["position_prev"] * bt["ret_1"] - bt["cost"]
    bt["equity"] = (1.0 + bt["strategy_ret"]).cumprod()
    bt["benchmark_equity"] = (1.0 + bt["ret_1"]).cumprod()

    bars_per_year = (60 * 24 * 365) / interval_to_minutes(args.interval)
    n_bars = len(bt)
    strat_total = float(bt["equity"].iloc[-1] - 1.0)
    bench_total = float(bt["benchmark_equity"].iloc[-1] - 1.0)
    ann_return = float((bt["equity"].iloc[-1]) ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else 0.0
    vol = float(bt["strategy_ret"].std(ddof=0) * math.sqrt(bars_per_year))
    sharpe = float(ann_return / vol) if vol > 0 else 0.0

    trade_stats = collect_trade_stats(bt)
    summary = {
        "symbol": args.symbol,
        "interval": args.interval,
        "start_time": str(bt["open_time"].iloc[0]),
        "end_time": str(bt["open_time"].iloc[-1]),
        "bars": int(n_bars),
        "buy_threshold": args.buy_threshold,
        "sell_threshold": args.sell_threshold,
        "position_mode": args.position_mode,
        "fee_bps": args.fee_bps,
        "slippage_bps": args.slippage_bps,
        "trades_turnover_count": int(bt["turnover"].sum()),
        "exposure_ratio": float(bt["position"].mean()),
        "strategy_total_return": strat_total,
        "benchmark_total_return": bench_total,
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(bt["equity"]),
        **trade_stats,
    }

    print("[4/4] writing outputs", flush=True)
    os.makedirs(os.path.dirname(args.summary_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.equity_out) or ".", exist_ok=True)
    with open(args.summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    bt[
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
