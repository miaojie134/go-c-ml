#!/usr/bin/env python3
import argparse
import json
import math
import os
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from backtest_lightgbm import (
    collect_trade_stats,
    interval_to_minutes,
    make_position,
    make_position_long_short,
    max_drawdown,
)
from train_lightgbm import add_features, load_kline_zip_files


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Threshold tuning on a fixed LightGBM model")
    p.add_argument("--data-root", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--symbol", required=True)
    p.add_argument("--interval", default="15m")
    p.add_argument("--trading-type", default="um", choices=["um", "cm"])
    p.add_argument("--time-period", default="daily", choices=["daily", "monthly"])
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--test-ratio", type=float, default=0.8)
    p.add_argument("--position-mode", default="long_short", choices=["long_only", "long_short"])
    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=5.0)

    p.add_argument("--buy-min", type=float, default=0.50)
    p.add_argument("--buy-max", type=float, default=0.70)
    p.add_argument("--buy-step", type=float, default=0.01)
    p.add_argument("--sell-min", type=float, default=0.10)
    p.add_argument("--sell-max", type=float, default=0.50)
    p.add_argument("--sell-step", type=float, default=0.01)

    p.add_argument("--min-return", type=float, default=0.0)
    p.add_argument("--min-trades-per-day", type=float, default=0.0)
    p.add_argument("--max-drawdown-limit", type=float, default=-1.0)
    p.add_argument("--optimize", default="balanced", choices=["return", "sharpe", "balanced"])
    p.add_argument(
        "--freq-weight",
        type=float,
        default=0.15,
        help="balanced score = return + freq_weight * trades_per_day",
    )

    p.add_argument("--results-out", required=True)
    p.add_argument("--best-out", required=True)
    return p.parse_args()


def frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


def choose_score(row: dict[str, Any], optimize: str, freq_weight: float) -> float:
    if optimize == "return":
        return float(row["strategy_total_return"])
    if optimize == "sharpe":
        return float(row["sharpe"])
    return float(row["strategy_total_return"] + freq_weight * row["trades_per_day"])


def main() -> None:
    args = parse_args()
    if not (0 < args.test_ratio < 1):
        raise ValueError("test-ratio must be in (0, 1)")
    if args.buy_step <= 0 or args.sell_step <= 0:
        raise ValueError("buy-step and sell-step must be > 0")

    print("[0/4] loading model", flush=True)
    booster = lgb.Booster(model_file=args.model_path)
    feature_names = booster.feature_name()

    print("[1/4] loading data + features", flush=True)
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

    feat = add_features(raw)
    missing = [c for c in feature_names if c not in feat.columns]
    if missing:
        raise RuntimeError(f"missing features: {missing[:10]}")
    feat = feat.dropna(subset=feature_names).reset_index(drop=True)
    if feat.empty:
        raise RuntimeError("no rows after feature dropna")

    split = int(len(feat) * args.test_ratio)
    bt = feat.iloc[split:].copy().reset_index(drop=True)
    if bt.empty:
        raise RuntimeError("backtest slice is empty")

    bt["ret_1"] = bt["close"].pct_change().fillna(0.0)
    bt["proba"] = booster.predict(bt[feature_names])

    bars_per_year = (60 * 24 * 365) / interval_to_minutes(args.interval)
    bars_per_day = (60 * 24) / interval_to_minutes(args.interval)
    cost_rate = (args.fee_bps + args.slippage_bps) / 10000.0

    buy_grid = frange(args.buy_min, args.buy_max, args.buy_step)
    sell_grid = frange(args.sell_min, args.sell_max, args.sell_step)
    combos = [(b, s) for b in buy_grid for s in sell_grid if s < b]
    if not combos:
        raise RuntimeError("no valid threshold combos")

    print(f"[2/4] evaluating {len(combos)} combos", flush=True)
    rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_score = -float("inf")

    for i, (buy_th, sell_th) in enumerate(combos, start=1):
        work = bt.copy()
        if args.position_mode == "long_short":
            work["position"] = make_position_long_short(work["proba"].to_numpy(), buy_th, sell_th)
        else:
            work["position"] = make_position(work["proba"].to_numpy(), buy_th, sell_th)
        work["position_prev"] = work["position"].shift(1).fillna(0).astype(int)
        work["turnover"] = (work["position"] - work["position_prev"]).abs()
        work["cost"] = work["turnover"] * cost_rate
        work["strategy_ret"] = work["position_prev"] * work["ret_1"] - work["cost"]
        work["equity"] = (1.0 + work["strategy_ret"]).cumprod()
        work["benchmark_equity"] = (1.0 + work["ret_1"]).cumprod()

        n_bars = len(work)
        ann_return = float((work["equity"].iloc[-1]) ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else 0.0
        vol = float(work["strategy_ret"].std(ddof=0) * math.sqrt(bars_per_year))
        sharpe = float(ann_return / vol) if vol > 0 else 0.0
        trade_stats = collect_trade_stats(work)
        days = n_bars / bars_per_day if bars_per_day > 0 else 0
        trades_per_day = float(trade_stats["closed_trades"] / days) if days > 0 else 0.0

        row = {
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
            "strategy_total_return": float(work["equity"].iloc[-1] - 1.0),
            "benchmark_total_return": float(work["benchmark_equity"].iloc[-1] - 1.0),
            "annualized_return": ann_return,
            "annualized_volatility": vol,
            "sharpe": sharpe,
            "max_drawdown": max_drawdown(work["equity"]),
            "trades_turnover_count": int(work["turnover"].sum()),
            "exposure_ratio": float(work["position"].mean()),
            "trades_per_day": trades_per_day,
            **trade_stats,
        }
        row["constraint_passed"] = bool(
            row["strategy_total_return"] >= args.min_return
            and row["trades_per_day"] >= args.min_trades_per_day
            and row["max_drawdown"] >= args.max_drawdown_limit
        )
        row["score"] = choose_score(row, args.optimize, args.freq_weight)
        rows.append(row)

        score = row["score"]
        if not row["constraint_passed"]:
            score = -float("inf")
        if score > best_score:
            best_score = score
            best_row = row

        if i % 100 == 0 or i == len(combos):
            print(f"[2/4] evaluated {i}/{len(combos)}", flush=True)

    print("[3/4] writing outputs", flush=True)
    results = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    if best_row is None:
        best_row = results.iloc[0].to_dict()
    payload = {
        "symbol": args.symbol,
        "interval": args.interval,
        "model_path": args.model_path,
        "position_mode": args.position_mode,
        "cost": {"fee_bps": args.fee_bps, "slippage_bps": args.slippage_bps},
        "optimize": args.optimize,
        "freq_weight": args.freq_weight,
        "constraints": {
            "min_return": args.min_return,
            "min_trades_per_day": args.min_trades_per_day,
            "max_drawdown_limit": args.max_drawdown_limit,
        },
        "constraint_passed_count": int(results["constraint_passed"].sum()),
        "best": best_row,
        "top10": results[results["constraint_passed"] == True].head(10).to_dict(orient="records"),
    }

    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_out) or ".", exist_ok=True)
    results.to_csv(args.results_out, index=False)
    with open(args.best_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("[4/4] done")
    print(f"best buy={best_row['buy_threshold']} sell={best_row['sell_threshold']}")
    print(f"best return={best_row['strategy_total_return']}, trades/day={best_row['trades_per_day']}, sharpe={best_row['sharpe']}")
    print(f"results saved: {args.results_out}")
    print(f"best saved: {args.best_out}")


if __name__ == "__main__":
    main()
