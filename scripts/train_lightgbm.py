#!/usr/bin/env python3
import argparse
import glob
import json
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


@dataclass
class Dataset:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    feature_names: list[str]
    raw: pd.DataFrame


def load_kline_zip_files(data_root: str, symbol: str, interval: str, trading_type: str, time_period: str) -> pd.DataFrame:
    pattern = os.path.join(
        data_root,
        "futures",
        trading_type,
        time_period,
        "klines",
        symbol.upper(),
        interval,
        f"{symbol.upper()}-{interval}-*.zip",
    )
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no zip files found: {pattern}")

    print(f"[1/4] loading {len(files)} zip files from: {os.path.dirname(pattern)}", flush=True)
    frames: list[pd.DataFrame] = []
    for i, f in enumerate(files, start=1):
        try:
            df = pd.read_csv(f, header=None)
            if df.shape[1] < len(KLINE_COLUMNS):
                print(f"[WARN] skip malformed file: {f}")
                continue
            df = df.iloc[:, : len(KLINE_COLUMNS)]
            df.columns = KLINE_COLUMNS
            frames.append(df)
            if i % 200 == 0 or i == len(files):
                print(f"[1/4] loaded {i}/{len(files)} files", flush=True)
        except Exception as e:
            print(f"[WARN] skip broken file: {f}, reason={e}")

    if not frames:
        raise RuntimeError("all zip files failed to read")

    out = pd.concat(frames, ignore_index=True)
    # Some Binance files include header-like text rows ("open_time", ...).
    # Coerce all numeric fields and drop invalid rows before timestamp parsing.
    for c in KLINE_COLUMNS:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    required = ["open_time", "open", "high", "low", "close", "volume"]
    before = len(out)
    out = out.dropna(subset=required).copy()
    dropped = before - len(out)
    if dropped > 0:
        print(f"[1/4] dropped {dropped} invalid rows", flush=True)

    out["open_time"] = pd.to_datetime(out["open_time"].astype("int64"), unit="ms", utc=True)
    out = out.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)
    print(f"[1/4] final rows: {len(out)}", flush=True)
    return out


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["ret_1"] = out["close"].pct_change(1)
    out["ret_2"] = out["close"].pct_change(2)
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_8"] = out["close"].pct_change(8)
    out["range_hl"] = (out["high"] - out["low"]) / out["close"]
    out["range_oc"] = (out["close"] - out["open"]) / out["open"]
    out["volume_chg_1"] = out["volume"].pct_change(1)
    out["trade_count_chg_1"] = out["number_of_trades"].pct_change(1)

    for w in (5, 10, 20, 48, 96):
        out[f"volatility_{w}"] = out["ret_1"].rolling(w).std()
        out[f"price_ma_ratio_{w}"] = out["close"] / out["close"].rolling(w).mean() - 1
        out[f"volume_ma_ratio_{w}"] = out["volume"] / out["volume"].rolling(w).mean() - 1

    try:
        import pandas_ta as ta

        out["ema_12"] = ta.ema(out["close"], length=12)
        out["ema_26"] = ta.ema(out["close"], length=26)
        out["ema_50"] = ta.ema(out["close"], length=50)
        out["sma_20"] = ta.sma(out["close"], length=20)
        out["sma_60"] = ta.sma(out["close"], length=60)
        out["rsi_14"] = ta.rsi(out["close"], length=14)
        out["rsi_28"] = ta.rsi(out["close"], length=28)
        out["atr_14"] = ta.atr(out["high"], out["low"], out["close"], length=14)

        macd = ta.macd(out["close"], fast=12, slow=26, signal=9)
        if macd is not None:
            out = out.join(macd)

        bbands = ta.bbands(out["close"], length=20, std=2)
        if bbands is not None:
            out = out.join(bbands)

        stoch = ta.stoch(out["high"], out["low"], out["close"], k=14, d=3, smooth_k=3)
        if stoch is not None:
            out = out.join(stoch)

        adx = ta.adx(out["high"], out["low"], out["close"], length=14)
        if adx is not None:
            out = out.join(adx)
    except Exception as e:
        print(f"[WARN] pandas-ta unavailable, continue with core features only: {e}")

    return out


def build_dataset(
    df: pd.DataFrame,
    horizon: int,
    train_ratio: float,
    start_date: str | None,
    end_date: str | None,
    target_threshold: float,
) -> Dataset:
    out = df.copy()
    if start_date:
        out = out[out["open_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        out = out[out["open_time"] <= pd.Timestamp(end_date, tz="UTC")]

    out = add_features(out)
    out["future_ret"] = out["close"].shift(-horizon) / out["close"] - 1
    out["target"] = (out["future_ret"] > target_threshold).astype(int)

    reserved_cols = {
        "open_time",
        "close_time",
        "future_ret",
        "target",
        "ignore",
    }
    feature_names = [
        c
        for c in out.columns
        if c not in reserved_cols and pd.api.types.is_numeric_dtype(out[c])
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
    y_train = train["target"].astype(int)
    X_test = test[feature_names]
    y_test = test["target"].astype(int)

    if y_train.nunique() < 2:
        raise RuntimeError("train set has only one class, adjust date range or target_threshold")
    if y_test.nunique() < 2:
        raise RuntimeError("test set has only one class, adjust date range or target_threshold")

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
) -> tuple[LGBMClassifier, dict]:
    model = LGBMClassifier(
        objective="binary",
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
        gpu_use_dp=gpu_use_dp,
        random_state=random_state,
        n_jobs=-1,
    )
    try:
        model.fit(
            ds.X_train,
            ds.y_train,
            eval_set=[(ds.X_test, ds.y_test)],
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=80), log_evaluation(100)],
        )
    except Exception as e:
        if device_type == "cpu":
            raise
        print(f"[WARN] device_type={device_type} failed ({e}), fallback to cpu", flush=True)
        model = LGBMClassifier(
            objective="binary",
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
            eval_metric="auc",
            callbacks=[early_stopping(stopping_rounds=80), log_evaluation(100)],
        )

    prob = model.predict_proba(ds.X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(ds.y_test, pred)),
        "precision": float(precision_score(ds.y_test, pred, zero_division=0)),
        "recall": float(recall_score(ds.y_test, pred, zero_division=0)),
        "f1": float(f1_score(ds.y_test, pred, zero_division=0)),
        "auc": float(roc_auc_score(ds.y_test, prob)),
        "train_samples": int(len(ds.X_train)),
        "test_samples": int(len(ds.X_test)),
        "n_features": int(len(ds.feature_names)),
    }
    return model, metrics


def save_outputs(
    model: LGBMClassifier,
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
