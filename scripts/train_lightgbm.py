#!/usr/bin/env python3
import argparse
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping, log_evaluation
from sklearn.metrics import mean_squared_error, r2_score

from ml_common import add_features, apply_date_filter, load_kline_zip_files


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    raw: pd.DataFrame


def build_dataset(
    df: pd.DataFrame,
    horizon: int,
    train_ratio: float,
    start_date: str | None,
    end_date: str | None,
    target_threshold: float,
) -> Dataset:
    out = apply_date_filter(df.copy(), start_date, end_date)
    out = add_features(out)
    out["future_ret"] = out["close"].shift(-horizon) / out["close"] - 1
    out["target"] = out["future_ret"]

    reserved_cols = {
        "open_time",
        "close_time",
        "future_ret",
        "target",
        "ignore",
        "tb_label",
    }
    noisy_cols = {"open", "high", "low", "close", "close_time", "open_time"}
    feature_names = [
        c
        for c in out.columns
        if c not in reserved_cols and c not in noisy_cols and pd.api.types.is_numeric_dtype(out[c])
    ]
    out = out.dropna(subset=feature_names + ["target"]).copy()
    if out.empty:
        raise RuntimeError("dataset empty after feature engineering and NaN drop")

    split = int(len(out) * train_ratio)
    if split <= 0 or split >= len(out):
        raise ValueError(f"invalid train_ratio={train_ratio}, samples={len(out)}")

    train = out.iloc[:split]
    test = out.iloc[split:]

    X_train = train[feature_names]
    y_train = train["target"]
    X_test = test[feature_names]
    y_test = test["target"]

    # For regression, we just need variance in target
    if y_train.nunique() <= 1:
        raise RuntimeError("train set target has no variance")
    if y_test.nunique() <= 1:
        raise RuntimeError("test set target has no variance")

    return Dataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        feature_names=feature_names,
        raw=out,
    )


def train_and_evaluate(
    ds: Dataset,
    random_state: int,
    device_type: str,
    gpu_platform_id: int,
    gpu_device_id: int,
    gpu_use_dp: int,
) -> tuple[LGBMRegressor, dict]:
    # LightGBM expects bool-like value for gpu_use_dp on newer versions.
    gpu_use_dp_flag = bool(gpu_use_dp)
    model = LGBMRegressor(
        objective="regression",
        n_estimators=1000,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        device_type=device_type,
        gpu_platform_id=gpu_platform_id,
        gpu_device_id=gpu_device_id,
        gpu_use_dp=gpu_use_dp_flag,
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        model.fit(
            ds.X_train,
            ds.y_train,
            eval_set=[(ds.X_test, ds.y_test)],
            eval_metric="l2",
            callbacks=[early_stopping(stopping_rounds=80), log_evaluation(100)],
        )
    except Exception as e:
        if device_type == "cpu":
            raise
        print(f"[WARN] device_type={device_type} failed ({e}), fallback to cpu", flush=True)
        model = LGBMRegressor(
            objective="regression",
            n_estimators=1000,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            device_type="cpu",
            random_state=random_state,
            n_jobs=-1,
        )
        model.fit(
            ds.X_train,
            ds.y_train,
            eval_set=[(ds.X_test, ds.y_test)],
            eval_metric="l2",
            callbacks=[early_stopping(stopping_rounds=80), log_evaluation(100)],
        )

    prob = model.predict(ds.X_test)

    metrics = {
        "rmse": float(np.sqrt(mean_squared_error(ds.y_test, prob))),
        "r2": float(r2_score(ds.y_test, prob)),
        "correlation": float(np.corrcoef(ds.y_test, prob)[0, 1]) if np.std(prob) > 0 else 0.0,
        "train_samples": int(len(ds.X_train)),
        "test_samples": int(len(ds.X_test)),
        "n_features": int(len(ds.feature_names)),
    }
    return model, metrics


def save_outputs(
    model: LGBMRegressor,
    metrics: dict,
    ds: Dataset,
    model_out: str,
    metrics_out: str,
    importance_out: str,
) -> None:
    os.makedirs(os.path.dirname(model_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(metrics_out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(importance_out) or ".", exist_ok=True)

    model.booster_.save_model(model_out)
    with open(metrics_out, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    imp = pd.DataFrame(
        {
            "feature": ds.feature_names,
            "importance_gain": model.booster_.feature_importance(importance_type="gain"),
            "importance_split": model.booster_.feature_importance(importance_type="split"),
        }
    ).sort_values("importance_gain", ascending=False)
    imp.to_csv(importance_out, index=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM using Binance Kline zip data")
    p.add_argument("--data-root", required=True, help="path to binance-public-data/python/data")
    p.add_argument("--symbol", default="ETHUSDT")
    p.add_argument("--interval", default="15m")
    p.add_argument("--trading-type", default="um", choices=["um", "cm"])
    p.add_argument("--time-period", default="daily", choices=["daily", "monthly"])
    p.add_argument("--start-date", default="2022-01-01")
    p.add_argument("--end-date", default=None)
    p.add_argument("--horizon", type=int, default=2, help="predict next N bars return > threshold")
    p.add_argument("--target-threshold", type=float, default=0.0, help="future return threshold")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--device-type", default="cuda", choices=["cpu", "gpu", "cuda"])
    p.add_argument("--gpu-platform-id", type=int, default=0)
    p.add_argument("--gpu-device-id", type=int, default=0)
    p.add_argument("--gpu-use-dp", type=int, default=0, choices=[0, 1])
    p.add_argument("--model-out", default="models/eth_quant_model.txt")
    p.add_argument("--metrics-out", default="models/metrics.json")
    p.add_argument("--importance-out", default="models/feature_importance.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print("[0/4] start training pipeline", flush=True)
    raw = load_kline_zip_files(
        data_root=args.data_root,
        symbol=args.symbol,
        interval=args.interval,
        trading_type=args.trading_type,
        time_period=args.time_period,
    )
    print("[2/4] building features and labels", flush=True)
    ds = build_dataset(
        df=raw,
        horizon=args.horizon,
        train_ratio=args.train_ratio,
        start_date=args.start_date,
        end_date=args.end_date,
        target_threshold=args.target_threshold,
    )
    print(
        f"[2/4] samples train={len(ds.X_train)}, test={len(ds.X_test)}, features={len(ds.feature_names)}",
        flush=True,
    )
    print("[3/4] training LightGBM", flush=True)
    model, metrics = train_and_evaluate(
        ds,
        random_state=args.random_state,
        device_type=args.device_type,
        gpu_platform_id=args.gpu_platform_id,
        gpu_device_id=args.gpu_device_id,
        gpu_use_dp=args.gpu_use_dp,
    )
    print("[4/4] saving outputs", flush=True)
    save_outputs(model, metrics, ds, args.model_out, args.metrics_out, args.importance_out)

    print("\n=== Train Done ===")
    for k, v in metrics.items():
        print(f"{k}: {v}")
    print(f"model saved: {args.model_out}")
    print(f"metrics saved: {args.metrics_out}")
    print(f"feature importance saved: {args.importance_out}")


if __name__ == "__main__":
    main()
