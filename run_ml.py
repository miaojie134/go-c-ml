#!/usr/bin/env python3
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import venv
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PROJECT_VENV_DIR = ROOT / ".venv"
BINANCE_DATA_BASE_URL = "https://data.binance.vision/data/futures"


def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _looks_like_venv_python(python_bin: Path) -> bool:
    parts = [p.lower() for p in python_bin.parts]
    return "scripts" in parts or "bin" in parts


def _running_in_venv() -> bool:
    return sys.prefix != getattr(sys, "base_prefix", sys.prefix)


def _resolve_default_venv_python() -> Path:
    env_venv_py = os.getenv("VENV_PY")
    if env_venv_py:
        return Path(env_venv_py)

    env_venv_dir = os.getenv("VENV_DIR")
    if env_venv_dir:
        return _venv_python_path(Path(env_venv_dir))

    candidates: list[Path] = []
    running_py = Path(sys.executable)
    if _running_in_venv() and running_py.is_file():
        candidates.append(running_py)

    candidates.extend(
        [
            _venv_python_path(PROJECT_VENV_DIR),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return _venv_python_path(PROJECT_VENV_DIR)


VENV_PY = _resolve_default_venv_python()
VENV_DIR = VENV_PY.parent.parent if _looks_like_venv_python(VENV_PY) else PROJECT_VENV_DIR
DATA_ROOT = Path(os.getenv("DATA_ROOT", str(ROOT / "data")))
INTERVAL = os.getenv("INTERVAL", "15m")
TRADING_TYPE = os.getenv("TRADING_TYPE", "um")
TIME_PERIOD = os.getenv("TIME_PERIOD", "daily")
START_DATE = os.getenv("START_DATE", "2023-01-01")
END_DATE = os.getenv("END_DATE", "")


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, env=env)


def python_major_minor(python_bin: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        [str(python_bin), "-c", "import sys;print(f'{sys.version_info.major}.{sys.version_info.minor}')"],
        text=True,
    ).strip()
    major, minor = out.split(".", 1)
    return int(major), int(minor)


def ensure_python() -> None:
    if VENV_PY.is_file():
        return
    host_py = shutil_which("python3") or shutil_which("python")
    if not host_py:
        raise SystemExit("python3/python not found, cannot create venv")
    print(f"[setup] creating venv: {VENV_DIR}", flush=True)
    venv.EnvBuilder(with_pip=True).create(str(VENV_DIR))


def ensure_deps() -> None:
    probe = "import lightgbm, pandas, numpy, sklearn"
    ok = subprocess.run([str(VENV_PY), "-c", probe], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    if ok == 0:
        return

    py_ver = python_major_minor(VENV_PY)
    if py_ver >= (3, 14):
        print(
            "[setup] Python >= 3.14 detected. pandas-ta will be skipped (optional), core training still works.",
            flush=True,
        )

    print("[setup] installing dependencies ...", flush=True)
    try:
        run([str(VENV_PY), "-m", "pip", "install", "-U", "pip"])
        run([str(VENV_PY), "-m", "pip", "install", "-r", "requirements.txt"])
    except subprocess.CalledProcessError as exc:
        if py_ver >= (3, 14):
            raise SystemExit(
                "dependency install failed on Python >= 3.14.\n"
                "Please use Python 3.10-3.13 virtualenv, or set VENV_PY to an existing project venv python."
            ) from exc
        raise


def ensure_runtime() -> None:
    ensure_python()
    ensure_deps()


def shutil_which(binary: str) -> str | None:
    path = os.environ.get("PATH", "")
    if not path:
        return None
    exts = [""]
    if os.name == "nt":
        pathext = os.environ.get("PATHEXT", ".EXE;.BAT;.CMD")
        exts = pathext.split(";")
    for p in path.split(os.pathsep):
        base = Path(p) / binary
        for ext in exts:
            cand = Path(str(base) + ext)
            if cand.is_file():
                return str(cand)
    return None


def require_symbol(command_name: str, args: list[str]) -> tuple[str, list[str]]:
    if not args or args[0].startswith("-"):
        print(f"error: SYMBOL is required for '{command_name}'", file=sys.stderr)
        print(f"example: python run_ml.py {command_name} ETHUSDT", file=sys.stderr)
        raise SystemExit(2)
    return args[0], args[1:]


def resolve_common_overrides(extra_args: list[str]) -> tuple[str, str, str, str, str]:
    interval = INTERVAL
    trading_type = TRADING_TYPE
    time_period = TIME_PERIOD
    start_date = START_DATE
    end_date = END_DATE

    i = 0
    while i < len(extra_args):
        token = extra_args[i]
        if token == "--interval" and i + 1 < len(extra_args):
            interval = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--interval="):
            interval = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--trading-type" and i + 1 < len(extra_args):
            trading_type = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--trading-type="):
            trading_type = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--time-period" and i + 1 < len(extra_args):
            time_period = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--time-period="):
            time_period = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--start-date" and i + 1 < len(extra_args):
            start_date = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--start-date="):
            start_date = token.split("=", 1)[1]
            i += 1
            continue
        if token == "--end-date" and i + 1 < len(extra_args):
            end_date = extra_args[i + 1]
            i += 2
            continue
        if token.startswith("--end-date="):
            end_date = token.split("=", 1)[1]
            i += 1
            continue
        i += 1

    return interval, trading_type, time_period, start_date, end_date


def has_kline_data(symbol_upper: str, interval: str, trading_type: str, time_period: str) -> bool:
    pattern = (
        DATA_ROOT
        / "futures"
        / trading_type
        / time_period
        / "klines"
        / symbol_upper
        / interval
        / f"{symbol_upper}-{interval}-*.zip"
    )
    return len(glob.glob(str(pattern))) > 0


def _parse_iso_date(label: str, value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise SystemExit(f"invalid {label}: {value}, expected YYYY-MM-DD") from exc


def _iter_month_starts(start_day: date, end_day: date):
    cur = date(start_day.year, start_day.month, 1)
    end = date(end_day.year, end_day.month, 1)
    while cur <= end:
        yield cur
        if cur.month == 12:
            cur = date(cur.year + 1, 1, 1)
        else:
            cur = date(cur.year, cur.month + 1, 1)


def _download_to_file(url: str, target_path: Path) -> str:
    if target_path.is_file():
        return "exists"

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    req = urlrequest.Request(url, headers={"User-Agent": "go-c-ml/1.0"})
    try:
        with urlrequest.urlopen(req, timeout=60) as resp, tmp_path.open("wb") as f:
            shutil.copyfileobj(resp, f)
        tmp_path.replace(target_path)
        return "downloaded"
    except urlerror.HTTPError as exc:
        if exc.code == 404:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            return "not_found"
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def ensure_kline_data(
    symbol: str,
    interval: str,
    trading_type: str,
    time_period: str,
    start_date: str,
    end_date: str,
) -> None:
    symbol_upper = symbol.upper()
    if has_kline_data(symbol_upper, interval, trading_type, time_period):
        return

    start_day = _parse_iso_date("start-date", start_date)
    end_day = _parse_iso_date("end-date", end_date) if end_date else datetime.now(timezone.utc).date()
    if start_day > end_day:
        raise SystemExit(f"start-date {start_date} is later than end-date {end_day.isoformat()}")

    save_dir = DATA_ROOT / "futures" / trading_type / time_period / "klines" / symbol_upper / interval
    save_dir.mkdir(parents=True, exist_ok=True)

    downloaded = 0
    exists = 0
    not_found = 0
    print(
        f"[setup] data not found, downloading klines from Binance: symbol={symbol_upper} interval={interval} type={trading_type} period={time_period}",
        flush=True,
    )

    if time_period == "daily":
        cur = start_day
        while cur <= end_day:
            day_str = cur.isoformat()
            file_name = f"{symbol_upper}-{interval}-{day_str}.zip"
            url = f"{BINANCE_DATA_BASE_URL}/{trading_type}/daily/klines/{symbol_upper}/{interval}/{file_name}"
            status = _download_to_file(url, save_dir / file_name)
            if status == "downloaded":
                downloaded += 1
            elif status == "exists":
                exists += 1
            else:
                not_found += 1
            cur += timedelta(days=1)
    elif time_period == "monthly":
        for month_start in _iter_month_starts(start_day, end_day):
            month_str = f"{month_start.year:04d}-{month_start.month:02d}"
            file_name = f"{symbol_upper}-{interval}-{month_str}.zip"
            url = f"{BINANCE_DATA_BASE_URL}/{trading_type}/monthly/klines/{symbol_upper}/{interval}/{file_name}"
            status = _download_to_file(url, save_dir / file_name)
            if status == "downloaded":
                downloaded += 1
            elif status == "exists":
                exists += 1
            else:
                not_found += 1
    else:
        raise SystemExit(f"unsupported time-period: {time_period}, expected daily or monthly")

    print(
        f"[setup] download summary: downloaded={downloaded}, existing={exists}, not_found={not_found}",
        flush=True,
    )

    if not has_kline_data(symbol_upper, interval, trading_type, time_period):
        raise SystemExit(
            f"download finished but still no data found for {symbol_upper} ({trading_type}/{time_period}/{interval})"
        )


def to_prefix(symbol: str) -> str:
    base = symbol[:-4] if symbol.endswith("USDT") else symbol
    if not base:
        base = symbol
    return base.lower()


def usage() -> str:
    return """Usage:
  python run_ml.py <command> SYMBOL [extra args...]

Commands:
  train         Train one model
  auto          Auto-search long_only config
  auto-ls       Auto-search long_short config
  backtest      Backtest quant model
  backtest-best Backtest best long_short model
  grid          Grid-search thresholds on quant model

Examples:
  python run_ml.py train ETHUSDT
  python run_ml.py train BTCUSDT --start-date 2022-01-01
  python run_ml.py auto-ls ETHUSDT
  python run_ml.py backtest ETHUSDT
"""


def run_train(symbol: str, extra_args: list[str]) -> None:
    ensure_runtime()
    interval, trading_type, time_period, start_date, end_date = resolve_common_overrides(extra_args)
    ensure_kline_data(symbol, interval, trading_type, time_period, start_date, end_date)
    prefix = to_prefix(symbol)
    print(f"[train] symbol={symbol} interval={interval} start_date={start_date} device=cuda out_prefix={prefix}", flush=True)
    cmd = [
        str(VENV_PY),
        "scripts/train_lightgbm.py",
        "--data-root",
        str(DATA_ROOT),
        "--symbol",
        symbol,
        "--interval",
        interval,
        "--trading-type",
        trading_type,
        "--time-period",
        time_period,
        "--start-date",
        start_date,
        "--model-out",
        f"./models/{prefix}_quant_model.txt",
        "--metrics-out",
        f"./models/{prefix}_metrics.json",
        "--importance-out",
        f"./models/{prefix}_feature_importance.csv",
    ]
    if end_date:
        cmd.extend(["--end-date", end_date])
    cmd.extend(extra_args)
    run(cmd)


def run_auto(symbol: str, extra_args: list[str], long_short: bool) -> None:
    ensure_runtime()
    interval, trading_type, time_period, start_date, end_date = resolve_common_overrides(extra_args)
    ensure_kline_data(symbol, interval, trading_type, time_period, start_date, end_date)
    prefix = to_prefix(symbol)
    mode = "long_short" if long_short else "long_only"
    print(f"[auto] symbol={symbol} mode={mode} interval={interval} start_date={start_date} device=cuda", flush=True)
    cmd = [
        str(VENV_PY),
        "scripts/auto_search_eth.py",
        "--data-root",
        str(DATA_ROOT),
        "--symbol",
        symbol,
        "--interval",
        interval,
        "--trading-type",
        trading_type,
        "--time-period",
        time_period,
        "--start-date",
        start_date,
        "--train-ratio",
        "0.7",
        "--val-ratio",
        "0.85",
        "--horizons",
        "1,2,3,4,6" if long_short else "1,2,3,4",
        "--target-thresholds",
        "0.0,0.0005,0.001",
        "--learning-rates",
        "0.02,0.05",
        "--num-leaves",
        "31,63,127",
        "--n-estimators",
        "700",
        "--early-stop-rounds",
        "80",
        "--buy-min",
        "0.52",
        "--buy-max",
        "0.80",
        "--buy-step",
        "0.02",
        "--sell-min",
        "0.20",
        "--sell-max",
        "0.48" if long_short else "0.50",
        "--sell-step",
        "0.02",
        "--fee-bps",
        "5",
        "--slippage-bps",
        "1",
        "--min-trades",
        "20" if long_short else "8",
        "--optimize",
        "strategy_total_return",
        "--results-out",
        f"./models/{prefix}_{'auto_search_ls_results' if long_short else 'auto_search_results'}.csv",
        "--best-json-out",
        f"./models/{prefix}_{'best_ls_config' if long_short else 'best_config'}.json",
        "--best-model-out",
        f"./models/{prefix}_{'best_ls_model' if long_short else 'best_model'}.txt",
        "--best-equity-out",
        f"./models/{prefix}_{'best_ls_equity' if long_short else 'best_equity'}.csv",
    ]
    if long_short:
        cmd.extend(["--position-mode", "long_short"])
    if end_date:
        cmd.extend(["--end-date", end_date])
    cmd.extend(extra_args)
    run(cmd)


def run_backtest(symbol: str, extra_args: list[str], best: bool) -> None:
    ensure_runtime()
    interval, trading_type, time_period, start_date, end_date = resolve_common_overrides(extra_args)
    ensure_kline_data(symbol, interval, trading_type, time_period, start_date, end_date)
    prefix = to_prefix(symbol)

    if best:
        best_model_refine = Path(f"./models/{prefix}_best_ls_refine_model.txt")
        best_model = best_model_refine if best_model_refine.is_file() else Path(f"./models/{prefix}_best_ls_model.txt")
        if not best_model.is_file():
            raise SystemExit(
                f"best model not found: ./models/{prefix}_best_ls_refine_model.txt or ./models/{prefix}_best_ls_model.txt\n"
                f"run auto-ls first: python run_ml.py auto-ls {symbol}"
            )
        print(f"[backtest] symbol={symbol} model={best_model} mode=long_short", flush=True)
        cmd = [
            str(VENV_PY),
            "scripts/backtest_lightgbm.py",
            "--data-root",
            str(DATA_ROOT),
            "--model-path",
            str(best_model),
            "--symbol",
            symbol,
            "--interval",
            interval,
            "--trading-type",
            trading_type,
            "--time-period",
            time_period,
            "--start-date",
            start_date,
            "--position-mode",
            "long_short",
            "--buy-threshold",
            "0.54",
            "--sell-threshold",
            "0.14",
            "--fee-bps",
            "5",
            "--slippage-bps",
            "1",
            "--summary-out",
            f"./models/{prefix}_best_live_summary.json",
            "--equity-out",
            f"./models/{prefix}_best_live_equity.csv",
        ]
    else:
        print(f"[backtest] symbol={symbol} model=./models/{prefix}_quant_model.txt", flush=True)
        cmd = [
            str(VENV_PY),
            "scripts/backtest_lightgbm.py",
            "--data-root",
            str(DATA_ROOT),
            "--model-path",
            f"./models/{prefix}_quant_model.txt",
            "--symbol",
            symbol,
            "--interval",
            interval,
            "--trading-type",
            trading_type,
            "--time-period",
            time_period,
            "--start-date",
            start_date,
            "--buy-threshold",
            "0.55",
            "--sell-threshold",
            "0.45",
            "--fee-bps",
            "5",
            "--slippage-bps",
            "1",
            "--summary-out",
            f"./models/{prefix}_backtest_summary.json",
            "--equity-out",
            f"./models/{prefix}_backtest_equity.csv",
        ]

    if end_date:
        cmd.extend(["--end-date", end_date])
    cmd.extend(extra_args)
    run(cmd)


def run_grid(symbol: str, extra_args: list[str]) -> None:
    ensure_runtime()
    interval, trading_type, time_period, start_date, end_date = resolve_common_overrides(extra_args)
    ensure_kline_data(symbol, interval, trading_type, time_period, start_date, end_date)
    prefix = to_prefix(symbol)
    print(f"[grid] symbol={symbol} model=./models/{prefix}_quant_model.txt", flush=True)
    cmd = [
        str(VENV_PY),
        "scripts/grid_search_backtest.py",
        "--data-root",
        str(DATA_ROOT),
        "--model-path",
        f"./models/{prefix}_quant_model.txt",
        "--symbol",
        symbol,
        "--interval",
        interval,
        "--trading-type",
        trading_type,
        "--time-period",
        time_period,
        "--start-date",
        start_date,
        "--test-ratio",
        "0.8",
        "--fee-bps",
        "5",
        "--slippage-bps",
        "1",
        "--buy-min",
        "0.52",
        "--buy-max",
        "0.72",
        "--buy-step",
        "0.02",
        "--sell-min",
        "0.30",
        "--sell-max",
        "0.50",
        "--sell-step",
        "0.02",
        "--min-trades",
        "20",
        "--optimize",
        "sharpe",
        "--top-k",
        "15",
        "--results-out",
        f"./models/{prefix}_grid_results.csv",
        "--best-out",
        f"./models/{prefix}_grid_best.json",
        "--best-equity-out",
        f"./models/{prefix}_grid_best_equity.csv",
    ]
    if end_date:
        cmd.extend(["--end-date", end_date])
    cmd.extend(extra_args)
    run(cmd)


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0

    cmd = argv[0]
    args = argv[1:]
    if cmd not in {"train", "auto", "auto-ls", "backtest", "backtest-best", "grid"}:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print()
        print(usage())
        return 1

    symbol, extra_args = require_symbol(cmd, args)

    if cmd == "train":
        run_train(symbol, extra_args)
    elif cmd == "auto":
        run_auto(symbol, extra_args, long_short=False)
    elif cmd == "auto-ls":
        run_auto(symbol, extra_args, long_short=True)
    elif cmd == "backtest":
        run_backtest(symbol, extra_args, best=False)
    elif cmd == "backtest-best":
        run_backtest(symbol, extra_args, best=True)
    elif cmd == "grid":
        run_grid(symbol, extra_args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
