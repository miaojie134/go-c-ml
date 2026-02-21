#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_lightgbm import collect_trade_stats, interval_to_minutes, make_position, max_drawdown
from train_lightgbm import add_features, load_kline_zip_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Grid search threshold backtest for LightGBM model")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--trading-type", default="um", choices=["um", "cm"])
    p.add_argument("--time-period", default="daily", choices=["daily", "monthly"])
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--test-ratio", type=float, default=0.8)
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)

    p.add_argument("--buy-min", type=float, default=0.52)
    p.add_argument("--buy-max", type=float, default=0.72)
    p.add_argument("--buy-step", type=float, default=0.02)
    p.add_argument("--sell-min", type=float, default=0.30)
    p.add_argument("--sell-max", type=float, default=0.50)
    p.add_argument("--sell-step", type=float, default=0.02)

    p.add_argument("--min-trades", type=int, default=20)
    p.add_argument("--optimize", default="sharpe", choices=["sharpe", "strategy_total_return", "annualized_return"])
    p.add_argument("--top-k", type=int, default=15)

    p.add_argument("--results-out", default="models/eth_grid_results.csv")
    p.add_argument("--best-out", default="models/eth_grid_best.json")
    p.add_argument("--best-equity-out", default="models/eth_grid_best_equity.csv")
    return p.parse_args()


def frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    values = [start + i * step for i in range(n + 1)]
    return [round(v, 10) for v in values]


def evaluate_combo(
    bt: pd.DataFrame,
    bars_per_year: float,
    buy_th: float,
    sell_th: float,
    cost_rate: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    work = bt.copy()
    work["position"] = make_position(work["proba"].to_numpy(), buy_th, sell_th)
    work["position_prev"] = work["position"].shift(1).fillna(0).astype(int)
    work["turnover"] = (work["position"] - work["position_prev"]).abs()
    work["cost"] = work["turnover"] * cost_rate
    work["strategy_ret"] = work["position_prev"] * work["ret_1"] - work["cost"]
    work["equity"] = (1.0 + work["strategy_ret"]).cumprod()
    work["benchmark_equity"] = (1.0 + work["ret_1"]).cumprod()

    n_bars = len(work)
    strat_total = float(work["equity"].iloc[-1] - 1.0)
    ann_return = float((work["equity"].iloc[-1]) ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else 0.0
    vol = float(work["strategy_ret"].std(ddof=0) * math.sqrt(bars_per_year))
    sharpe = float(ann_return / vol) if vol > 0 else 0.0
    trade_stats = collect_trade_stats(work)

    row = {
        "buy_threshold": buy_th,
        "sell_threshold": sell_th,
        "trades_turnover_count": int(work["turnover"].sum()),
        "exposure_ratio": float(work["position"].mean()),
        "strategy_total_return": strat_total,
        "benchmark_total_return": float(work["benchmark_equity"].iloc[-1] - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(work["equity"]),
        **trade_stats,
    }
    return row, work


def main() -> None:
    args = parse_args()
    if not (0 < args.test_ratio < 1):
        raise ValueError("test-ratio must be in (0, 1)")
    if args.buy_step <= 0 or args.sell_step <= 0:
        raise ValueError("steps must be > 0")

    print("[0/5] loading model", flush=True)
    booster = lgb.Booster(model_file=args.model_path)
    feature_names = booster.feature_name()

    print("[1/5] loading data", flush=True)
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

    print("[2/5] building features", flush=True)
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
    bt["ret_1"] = bt["close"].pct_change().fillna(0.0)
    bt["proba"] = booster.predict(bt[feature_names])

    bars_per_year = (60 * 24 * 365) / interval_to_minutes(args.interval)
    cost_rate = (args.fee_bps + args.slippage_bps) / 10000.0

    buy_grid = frange(args.buy_min, args.buy_max, args.buy_step)
    sell_grid = frange(args.sell_min, args.sell_max, args.sell_step)
    combos = [(b, s) for b in buy_grid for s in sell_grid if s < b]
    if not combos:
        raise RuntimeError("no valid buy/sell threshold combos generated")

    print(f"[3/5] evaluating {len(combos)} combos", flush=True)
    rows: list[dict[str, Any]] = []
    best_equity = None
    best_score = -float("inf")
    best_row: dict[str, Any] | None = None

    for i, (buy_th, sell_th) in enumerate(combos, start=1):
        row, equity = evaluate_combo(bt, bars_per_year, buy_th, sell_th, cost_rate)
        rows.append(row)
        if row["closed_trades"] >= args.min_trades and row[args.optimize] > best_score:
            best_score = row[args.optimize]
            best_row = row
            best_equity = equity
        if i % 20 == 0 or i == len(combos):
            print(f"[3/5] evaluated {i}/{len(combos)}", flush=True)

    results = pd.DataFrame(rows).sort_values(args.optimize, ascending=False).reset_index(drop=True)

    if best_row is None:
        best_idx = int(results["closed_trades"].idxmax())
        best_row = results.iloc[best_idx].to_dict()
        best_equity = evaluate_combo(
            bt,
            bars_per_year,
            float(best_row["buy_threshold"]),
            float(best_row["sell_threshold"]),
            cost_rate,
        )[1]

    top_source = results[results["closed_trades"] >= args.min_trades]
    if top_source.empty:
        top_source = results

    best_payload = {
        "symbol": args.symbol,
        "interval": args.interval,
        "optimize_metric": args.optimize,
        "min_trades": args.min_trades,
        "search_space": {
            "buy": [args.buy_min, args.buy_max, args.buy_step],
            "sell": [args.sell_min, args.sell_max, args.sell_step],
        },
        "best": best_row,
        "top": top_source.head(args.top_k).to_dict(orient="records"),
    }

    print("[4/5] writing outputs", flush=True)
    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_equity_out) or ".", exist_ok=True)
    results.to_csv(args.results_out, index=False)
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(best_payload, f, indent=2, ensure_ascii=False)
    best_equity[
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
    ].to_csv(args.best_equity_out, index=False)

    print("[5/5] done")
    print(f"best buy={best_row['buy_threshold']}, sell={best_row['sell_threshold']}")
    print(f"{args.optimize}={best_row[args.optimize]}")
    print(f"results saved: {args.results_out}")
    print(f"best summary saved: {args.best_out}")
    print(f"best equity saved: {args.best_equity_out}")


if __name__ == "__main__":
    main()
