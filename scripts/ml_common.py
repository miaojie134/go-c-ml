#!/usr/bin/env python3
from __future__ import annotations

import glob
import math
import os
from typing import Any

import numpy as np
import pandas as pd

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
    for i, path in enumerate(files, start=1):
        try:
            df = pd.read_csv(path, header=None)
            if df.shape[1] < len(KLINE_COLUMNS):
                print(f"[WARN] skip malformed file: {path}")
                continue
            df = df.iloc[:, : len(KLINE_COLUMNS)]
            df.columns = KLINE_COLUMNS
            frames.append(df)
            if i % 200 == 0 or i == len(files):
                print(f"[1/4] loaded {i}/{len(files)} files", flush=True)
        except Exception as exc:
            print(f"[WARN] skip broken file: {path}, reason={exc}")

    if not frames:
        raise RuntimeError("all zip files failed to read")

    out = pd.concat(frames, ignore_index=True)
    for col in KLINE_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

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


def apply_date_filter(df: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    out = df
    if start_date:
        out = out[out["open_time"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        out = out[out["open_time"] <= pd.Timestamp(end_date, tz="UTC")]
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

    for window in (5, 10, 20, 48, 96):
        out[f"volatility_{window}"] = out["ret_1"].rolling(window).std()
        out[f"price_ma_ratio_{window}"] = out["close"] / out["close"].rolling(window).mean() - 1
        out[f"volume_ma_ratio_{window}"] = out["volume"] / out["volume"].rolling(window).mean() - 1
        
        roll_close_mean = out["close"].rolling(window).mean()
        roll_close_std = out["close"].rolling(window).std()
        out[f"zscore_close_{window}"] = (out["close"] - roll_close_mean) / (roll_close_std + 1e-8)
        
        roll_vol_mean = out["volume"].rolling(window).mean()
        roll_vol_std = out["volume"].rolling(window).std()
        out[f"zscore_volume_{window}"] = (out["volume"] - roll_vol_mean) / (roll_vol_std + 1e-8)

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
    except Exception as exc:
        print(f"[WARN] pandas-ta unavailable, continue with core features only: {exc}")

    return out


def frange(start: float, stop: float, step: float) -> list[float]:
    n = int(round((stop - start) / step))
    return [round(start + i * step, 10) for i in range(n + 1)]


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
    drawdown = equity / roll_max - 1.0
    return float(drawdown.min())


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
    # Count both long and short round-trips. A trade starts when position moves
    # from 0 -> non-zero and ends when it goes back to 0 or flips sign.
    positions = df["position"].astype(int).to_numpy()
    idx = df.index.to_numpy()
    trade_ranges: list[tuple[Any, Any]] = []

    state = 0
    entry = None
    for i, pos in enumerate(positions):
        if state == 0:
            if pos != 0:
                state = int(pos)
                entry = idx[i]
            continue

        if pos == 0 or int(pos) != state:
            trade_ranges.append((entry, idx[i]))
            if pos == 0:
                state = 0
                entry = None
            else:
                state = int(pos)
                entry = idx[i]

    if state != 0 and entry is not None:
        trade_ranges.append((entry, idx[-1]))

    trade_returns = []
    for ent, ex in trade_ranges:
        if ex <= ent:
            continue
        seg = df.loc[ent:ex]
        trade_returns.append(float((1.0 + seg["strategy_ret"]).prod() - 1.0))

    if not trade_returns:
        return {
            "closed_trades": 0,
            "win_rate": 0.0,
            "avg_trade_return": 0.0,
            "median_trade_return": 0.0,
        }

    wins = sum(1 for ret in trade_returns if ret > 0)
    return {
        "closed_trades": int(len(trade_returns)),
        "win_rate": float(wins / len(trade_returns)),
        "avg_trade_return": float(np.mean(trade_returns)),
        "median_trade_return": float(np.median(trade_returns)),
    }


def simulate_strategy(
    frame: pd.DataFrame,
    buy_th: float,
    sell_th: float,
    cost_rate: float,
    position_mode: str,
) -> pd.DataFrame:
    work = frame.copy()
    if position_mode == "long_short":
        work["position"] = make_position_long_short(work["proba"].to_numpy(), buy_th, sell_th)
    elif position_mode == "long_only":
        work["position"] = make_position(work["proba"].to_numpy(), buy_th, sell_th)
    else:
        raise ValueError(f"unsupported position_mode: {position_mode}")
    work["position_prev"] = work["position"].shift(1).fillna(0).astype(int)
    work["turnover"] = (work["position"] - work["position_prev"]).abs()
    work["cost"] = work["turnover"] * cost_rate
    work["strategy_ret"] = work["position_prev"] * work["ret_1"] - work["cost"]
    work["equity"] = (1.0 + work["strategy_ret"]).cumprod()
    work["benchmark_equity"] = (1.0 + work["ret_1"]).cumprod()
    return work


def summarize_strategy(work: pd.DataFrame, bars_per_year: float) -> dict[str, float]:
    n_bars = len(work)
    ann_return = float((work["equity"].iloc[-1]) ** (bars_per_year / n_bars) - 1.0) if n_bars > 0 else 0.0
    vol = float(work["strategy_ret"].std(ddof=0) * math.sqrt(bars_per_year))
    sharpe = float(ann_return / vol) if vol > 0 else 0.0
    return {
        "strategy_total_return": float(work["equity"].iloc[-1] - 1.0),
        "benchmark_total_return": float(work["benchmark_equity"].iloc[-1] - 1.0),
        "annualized_return": ann_return,
        "annualized_volatility": vol,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown(work["equity"]),
    }
