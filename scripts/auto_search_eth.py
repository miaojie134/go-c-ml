#!/usr/bin/env python3
import argparse
import itertools
import json
import os
import shutil
from typing import Any

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping

from ml_common import (
    add_features,
    apply_date_filter,
    collect_trade_stats,
    frange,
    interval_to_minutes,
    load_kline_zip_files,
    simulate_strategy,
    summarize_strategy,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Auto search best LightGBM model + thresholds")
    p.add_argument("--data-root", required=True)
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--trading-type", default="um", choices=["um", "cm"])
    p.add_argument("--time-period", default="daily", choices=["daily", "monthly"])
    p.add_argument("--start-date", default="2023-01-01")
    p.add_argument("--end-date", default=None)

    p.add_argument("--train-ratio", type=float, default=0.7)
    p.add_argument("--val-ratio", type=float, default=0.85)

    p.add_argument("--horizons", default="1,2,3,4")
    p.add_argument("--target-thresholds", default="0.0,0.0005,0.001")
    p.add_argument("--learning-rates", default="0.02,0.05")
    p.add_argument("--num-leaves", default="31,63,127")

    p.add_argument("--n-estimators", type=int, default=700)
    p.add_argument("--early-stop-rounds", type=int, default=80)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample-bytree", type=float, default=0.8)
    p.add_argument("--reg-alpha", type=float, default=0.1)
    p.add_argument("--reg-lambda", type=float, default=0.1)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device-type", default="cuda", choices=["cpu", "gpu", "cuda"])
    p.add_argument("--gpu-platform-id", type=int, default=0)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--gpu-use-dp", type=int, default=0, choices=[0, 1])

    p.add_argument("--buy-min", type=float, default=0.52)
    p.add_argument("--buy-max", type=float, default=0.80)
    p.add_argument("--buy-step", type=float, default=0.02)
    p.add_argument("--sell-min", type=float, default=0.20)
    p.add_argument("--sell-max", type=float, default=0.50)
    p.add_argument("--sell-step", type=float, default=0.02)

    p.add_argument("--fee-bps", type=float, default=5.0)
    p.add_argument("--slippage-bps", type=float, default=1.0)
    p.add_argument("--min-trades", type=int, default=8)
    p.add_argument("--min-trades-per-day", type=float, default=0.0)
    p.add_argument("--min-sharpe", type=float, default=-999.0)
    p.add_argument("--max-drawdown-limit", type=float, default=-1.0, help="minimum acceptable max_drawdown, e.g. -0.25")
    p.add_argument("--min-return", type=float, default=-1.0, help="minimum acceptable strategy_total_return")
    p.add_argument("--optimize", default="strategy_total_return", choices=["strategy_total_return", "annualized_return", "sharpe"])
    p.add_argument("--position-mode", default="long_only", choices=["long_only", "long_short"])

    p.add_argument("--results-out", default="models/eth_auto_search_results.csv")
    p.add_argument("--best-json-out", default="models/eth_best_config.json")
    p.add_argument("--best-model-out", default="models/eth_best_model.txt")
    p.add_argument("--best-equity-out", default="models/eth_best_equity.csv")
    return p.parse_args()


def parse_list(value: str, cast):
    return [cast(x.strip()) for x in value.split(",") if x.strip()]


def build_labeled_frame(feat_base: pd.DataFrame, horizon: int, target_threshold: float) -> tuple[pd.DataFrame, list[str]]:
    df = feat_base.copy()
    df["future_ret"] = df["close"].shift(-horizon) / df["close"] - 1.0
    df["target"] = (df["future_ret"] > target_threshold).astype(int)

    reserved_cols = {"open_time", "close_time", "future_ret", "target", "ignore"}
    feature_names = [c for c in df.columns if c not in reserved_cols and pd.api.types.is_numeric_dtype(df[c])]
    df = df.dropna(subset=feature_names + ["target"]).copy().reset_index(drop=True)
    return df, feature_names


def split_df(df: pd.DataFrame, train_ratio: float, val_ratio: float):
    n = len(df)
    i_train = int(n * train_ratio)
    i_val = int(n * val_ratio)
    if not (0 < i_train < i_val < n):
        raise ValueError(f"invalid split indices: {i_train}, {i_val}, n={n}")
    return df.iloc[:i_train].copy(), df.iloc[i_train:i_val].copy(), df.iloc[i_val:].copy()


def summarize_classification(y_true: pd.Series, proba: np.ndarray) -> dict[str, float]:
    pred = (proba >= 0.5).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) else 0.0
    return {
        "clf_accuracy": float(acc),
        "clf_precision": float(precision),
        "clf_recall": float(recall),
    }


def evaluate_thresholds(
    frame: pd.DataFrame,
    buy_grid: list[float],
    sell_grid: list[float],
    cost_rate: float,
    bars_per_year: float,
    bars_per_day: float,
    optimize: str,
    min_trades: int,
    min_trades_per_day: float,
    min_sharpe: float,
    max_drawdown_limit: float,
    min_return: float,
    position_mode: str,
) -> tuple[dict[str, Any], pd.DataFrame]:
    best = None
    best_df = None
    best_score = -float("inf")
    fallback = None
    fallback_df = None
    fallback_score = -float("inf")

    for buy_th in buy_grid:
        for sell_th in sell_grid:
            if sell_th >= buy_th:
                continue

            work = simulate_strategy(
                frame=frame,
                buy_th=buy_th,
                sell_th=sell_th,
                cost_rate=cost_rate,
                position_mode=position_mode,
            )
            stats = summarize_strategy(work, bars_per_year)
            trade_stats = collect_trade_stats(work)
            days = len(work) / bars_per_day if bars_per_day > 0 else 0
            trades_per_day = float(trade_stats["closed_trades"] / days) if days > 0 else 0.0
            row = {
                "buy_threshold": buy_th,
                "sell_threshold": sell_th,
                "trades_turnover_count": int(work["turnover"].sum()),
                "exposure_ratio": float(work["position"].mean()),
                "trades_per_day": trades_per_day,
                **stats,
                **trade_stats,
            }

            score = row[optimize]
            if score > fallback_score:
                fallback_score = score
                fallback = row
                fallback_df = work

            if (
                row["closed_trades"] < min_trades
                or row["trades_per_day"] < min_trades_per_day
                or row["sharpe"] < min_sharpe
                or row["max_drawdown"] < max_drawdown_limit
                or row["strategy_total_return"] < min_return
            ):
                score = -float("inf")

            if score > best_score:
                best_score = score
                best = row
                best_df = work

    if best is None:
        if fallback is None:
            raise RuntimeError("threshold search failed")
        return fallback, fallback_df
    return best, best_df


def main() -> None:
    args = parse_args()
    if not (0 < args.train_ratio < args.val_ratio < 1):
        raise ValueError("require 0 < train-ratio < val-ratio < 1")
    if args.buy_step <= 0 or args.sell_step <= 0:
        raise ValueError("buy-step/sell-step must be > 0")

    horizons = parse_list(args.horizons, int)
    target_thresholds = parse_list(args.target_thresholds, float)
    learning_rates = parse_list(args.learning_rates, float)
    num_leaves_grid = parse_list(args.num_leaves, int)
    buy_grid = frange(args.buy_min, args.buy_max, args.buy_step)
    sell_grid = frange(args.sell_min, args.sell_max, args.sell_step)
    cost_rate = (args.fee_bps + args.slippage_bps) / 10000.0

    print("[0/6] loading base data", flush=True)
    raw = load_kline_zip_files(
        data_root=args.data_root,
        symbol=args.symbol,
        interval=args.interval,
        trading_type=args.trading_type,
        time_period=args.time_period,
    )
    raw = apply_date_filter(raw, args.start_date, args.end_date)
    feat_base = add_features(raw)

    bars_per_year = (60 * 24 * 365) / interval_to_minutes(args.interval)
    bars_per_day = (60 * 24) / interval_to_minutes(args.interval)
    model_cfgs = list(itertools.product(horizons, target_thresholds, learning_rates, num_leaves_grid))
    print(f"[1/6] model configs: {len(model_cfgs)}", flush=True)

    rows = []
    best_score = -float("inf")
    best_row = None
    best_equity = None
    best_any_score = -float("inf")
    best_any_row = None
    best_any_equity = None

    for idx, (horizon, tgt, lr, leaves) in enumerate(model_cfgs, start=1):
        df, feature_names = build_labeled_frame(feat_base, horizon, tgt)
        if len(df) < 3000:
            continue

        train_df, val_df, test_df = split_df(df, args.train_ratio, args.val_ratio)
        X_train = train_df[feature_names]
        y_train = train_df["target"].astype(int)
        X_val = val_df[feature_names]
        y_val = val_df["target"].astype(int)
        X_test = test_df[feature_names]
        y_test = test_df["target"].astype(int)
        if y_train.nunique() < 2 or y_val.nunique() < 2 or y_test.nunique() < 2:
            continue

        model = LGBMClassifier(
            objective="binary",
            n_estimators=args.n_estimators,
            learning_rate=lr,
            num_leaves=leaves,
            subsample=args.subsample,
            colsample_bytree=args.colsample_bytree,
            reg_alpha=args.reg_alpha,
            reg_lambda=args.reg_lambda,
            device_type=args.device_type,
            gpu_platform_id=args.gpu_platform_id,
            gpu_device_id=args.gpu_device_id,
            gpu_use_dp=args.gpu_use_dp,
            verbosity=-1,
            random_state=args.random_state,
            n_jobs=-1,
        )
        try:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[early_stopping(stopping_rounds=args.early_stop_rounds, verbose=False)],
            )
        except Exception as exc:
            if args.device_type == "cpu":
                raise
            print(
                f"[WARN] {args.symbol} config {idx}: device_type={args.device_type} failed ({exc}), fallback to cpu",
                flush=True,
            )
            model = LGBMClassifier(
                objective="binary",
                n_estimators=args.n_estimators,
                learning_rate=lr,
                num_leaves=leaves,
                subsample=args.subsample,
                colsample_bytree=args.colsample_bytree,
                reg_alpha=args.reg_alpha,
                reg_lambda=args.reg_lambda,
                device_type="cpu",
                verbosity=-1,
                random_state=args.random_state,
                n_jobs=-1,
            )
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                eval_metric="auc",
                callbacks=[early_stopping(stopping_rounds=args.early_stop_rounds, verbose=False)],
            )

        val_bt = val_df[["open_time", "close"]].copy()
        val_bt["ret_1"] = val_bt["close"].pct_change().fillna(0.0)
        val_bt["proba"] = model.predict_proba(X_val)[:, 1]
        val_best, _ = evaluate_thresholds(
            frame=val_bt,
            buy_grid=buy_grid,
            sell_grid=sell_grid,
            cost_rate=cost_rate,
            bars_per_year=bars_per_year,
            bars_per_day=bars_per_day,
            optimize=args.optimize,
            min_trades=args.min_trades,
            min_trades_per_day=args.min_trades_per_day,
            min_sharpe=args.min_sharpe,
            max_drawdown_limit=args.max_drawdown_limit,
            min_return=args.min_return,
            position_mode=args.position_mode,
        )

        test_bt = test_df[["open_time", "close"]].copy()
        test_bt["ret_1"] = test_bt["close"].pct_change().fillna(0.0)
        test_bt["proba"] = model.predict_proba(X_test)[:, 1]

        buy_th = float(val_best["buy_threshold"])
        sell_th = float(val_best["sell_threshold"])
        work = simulate_strategy(
            frame=test_bt,
            buy_th=buy_th,
            sell_th=sell_th,
            cost_rate=cost_rate,
            position_mode=args.position_mode,
        )

        days = len(work) / bars_per_day if bars_per_day > 0 else 0
        stats = summarize_strategy(work, bars_per_year)
        trade_stats = collect_trade_stats(work)
        trades_per_day = float(trade_stats["closed_trades"] / days) if days > 0 else 0.0
        clf_stats = summarize_classification(y_test, test_bt["proba"].to_numpy())
        constraint_passed = (
            trade_stats["closed_trades"] >= args.min_trades
            and trades_per_day >= args.min_trades_per_day
            and stats["sharpe"] >= args.min_sharpe
            and stats["max_drawdown"] >= args.max_drawdown_limit
            and stats["strategy_total_return"] >= args.min_return
        )

        row = {
            "horizon": horizon,
            "target_threshold": tgt,
            "learning_rate": lr,
            "num_leaves": leaves,
            "best_iteration": int(model.best_iteration_ or args.n_estimators),
            "n_features": int(len(feature_names)),
            "train_samples": int(len(X_train)),
            "val_samples": int(len(X_val)),
            "test_samples": int(len(X_test)),
            "buy_threshold": buy_th,
            "sell_threshold": sell_th,
            "val_strategy_total_return": float(val_best["strategy_total_return"]),
            "val_sharpe": float(val_best["sharpe"]),
            "trades_turnover_count": int(work["turnover"].sum()),
            "exposure_ratio": float(work["position"].mean()),
            "trades_per_day": trades_per_day,
            "constraint_passed": bool(constraint_passed),
            **stats,
            **trade_stats,
            **clf_stats,
        }
        rows.append(row)

        any_score = row[args.optimize]
        if any_score > best_any_score:
            best_any_score = any_score
            best_any_row = row
            best_any_equity = work.copy()
            model.booster_.save_model(args.best_model_out + ".any")

        score = row[args.optimize] if constraint_passed else -float("inf")
        if score > best_score:
            best_score = score
            best_row = row
            best_equity = work.copy()
            model.booster_.save_model(args.best_model_out)

        if idx % 5 == 0 or idx == len(model_cfgs):
            print(f"[2/6] trained {idx}/{len(model_cfgs)} configs", flush=True)

    if not rows:
        raise RuntimeError("no valid result rows produced")

    print("[3/6] ranking results", flush=True)
    results = pd.DataFrame(rows).sort_values(args.optimize, ascending=False).reset_index(drop=True)

    if best_row is None:
        best_row = best_any_row if best_any_row is not None else results.iloc[0].to_dict()
        best_equity = best_any_equity if best_any_equity is not None else pd.DataFrame()
        if os.path.exists(args.best_model_out + ".any"):
            shutil.copyfile(args.best_model_out + ".any", args.best_model_out)

    pass_count = int(results["constraint_passed"].sum()) if "constraint_passed" in results.columns else 0
    top_source = results[results["constraint_passed"] == True] if "constraint_passed" in results.columns else results
    if top_source.empty:
        top_source = results

    payload = {
        "symbol": args.symbol,
        "interval": args.interval,
        "objective": args.optimize,
        "constraints": {
            "min_trades": args.min_trades,
            "min_trades_per_day": args.min_trades_per_day,
            "min_sharpe": args.min_sharpe,
            "max_drawdown_limit": args.max_drawdown_limit,
            "min_return": args.min_return,
        },
        "data_split": {"train_ratio": args.train_ratio, "val_ratio": args.val_ratio},
        "constraint_passed_count": pass_count,
        "search_space": {
            "position_mode": args.position_mode,
            "device_type": args.device_type,
            "gpu_platform_id": args.gpu_platform_id,
            "gpu_device_id": args.gpu_device_id,
            "gpu_use_dp": args.gpu_use_dp,
            "horizons": horizons,
            "target_thresholds": target_thresholds,
            "learning_rates": learning_rates,
            "num_leaves": num_leaves_grid,
            "buy": [args.buy_min, args.buy_max, args.buy_step],
            "sell": [args.sell_min, args.sell_max, args.sell_step],
        },
        "best": best_row,
        "top10": top_source.head(10).to_dict(orient="records"),
    }

    print("[4/6] writing outputs", flush=True)
    os.makedirs(os.path.dirname(args.results_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_json_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.best_equity_out) or ".", exist_ok=True)
    results.to_csv(args.results_out, index=False)
    with open(args.best_json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if not best_equity.empty:
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
    if os.path.exists(args.best_model_out + ".any"):
        os.remove(args.best_model_out + ".any")

    print("[5/6] done")
    print(
        f"best config: horizon={best_row['horizon']}, target_threshold={best_row['target_threshold']}, "
        f"lr={best_row['learning_rate']}, leaves={best_row['num_leaves']}"
    )
    print(f"best thresholds: buy={best_row['buy_threshold']}, sell={best_row['sell_threshold']}")
    print(f"{args.optimize}: {best_row[args.optimize]}")
    print(f"results saved: {args.results_out}")
    print(f"best config saved: {args.best_json_out}")
    print(f"best model saved: {args.best_model_out}")
    print(f"best equity saved: {args.best_equity_out}")


if __name__ == "__main__":
    main()
