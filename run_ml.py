#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import sys
import venv
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib import error as urlerror
from urllib import request as urlrequest


ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)

PROJECT_VENV_DIR = ROOT / ".venv"
BINANCE_DATA_BASE_URL = "https://data.binance.vision/data/futures"


def _venv_python_path(venv_dir: Path) -> Path:
    return venv_dir / "bin/python"


def _looks_like_venv_python(python_bin: Path) -> bool:
    parts = [p.lower() for p in python_bin.parts]
    return "bin" in parts


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
DEFAULT_CONFIG_PATH = ROOT / "run_ml.config.json"


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


def ensure_supported_platform() -> None:
    if os.name == "nt":
        raise SystemExit("native Windows is not supported. please run in WSL2/Linux")


def shutil_which(binary: str) -> str | None:
    path = os.environ.get("PATH", "")
    if not path:
        return None
    for p in path.split(os.pathsep):
        base = Path(p) / binary
        if base.is_file():
            return str(base)
    return None


def _extract_config_arg(args: list[str]) -> tuple[Path | None, list[str]]:
    out: list[str] = []
    config_path: Path | None = None
    i = 0
    while i < len(args):
        token = args[i]
        if token == "--config":
            if i + 1 >= len(args):
                raise SystemExit("missing value for --config")
            config_path = Path(args[i + 1])
            i += 2
            continue
        if token.startswith("--config="):
            config_path = Path(token.split("=", 1)[1])
            i += 1
            continue
        out.append(token)
        i += 1
    return config_path, out


def _load_config(config_path: Path | None) -> dict:
    path = config_path
    if path is None and DEFAULT_CONFIG_PATH.is_file():
        path = DEFAULT_CONFIG_PATH
    if path is None:
        return {}
    if not path.is_file():
        raise SystemExit(f"config file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"invalid JSON config: {path} ({exc})") from exc


COMMON_CONFIG_KEYS = {"symbol", "interval", "trading-type", "time-period", "start-date", "end-date"}
SUPPORTED_COMMANDS = ("data", "train", "auto", "auto-ls", "backtest", "backtest-best", "grid")
REQUIRED_COMMAND_KEYS: dict[str, tuple[str, ...]] = {
    "train": (
        "horizon",
        "target-threshold",
        "train-ratio",
        "device-type",
        "gpu-platform-id",
        "gpu-device-id",
        "gpu-use-dp",
    ),
    "auto": (
        "train-ratio",
        "val-ratio",
        "horizons",
        "target-thresholds",
        "learning-rates",
        "num-leaves",
        "n-estimators",
        "early-stop-rounds",
        "subsample",
        "colsample-bytree",
        "reg-alpha",
        "reg-lambda",
        "random-state",
        "device-type",
        "gpu-platform-id",
        "gpu-device-id",
        "gpu-use-dp",
        "buy-min",
        "buy-max",
        "buy-step",
        "sell-min",
        "sell-max",
        "sell-step",
        "fee-bps",
        "slippage-bps",
        "min-trades",
        "min-trades-per-day",
        "min-sharpe",
        "max-drawdown-limit",
        "min-return",
        "optimize",
        "position-mode",
    ),
    "auto-ls": (
        "train-ratio",
        "val-ratio",
        "horizons",
        "target-thresholds",
        "learning-rates",
        "num-leaves",
        "n-estimators",
        "early-stop-rounds",
        "subsample",
        "colsample-bytree",
        "reg-alpha",
        "reg-lambda",
        "random-state",
        "device-type",
        "gpu-platform-id",
        "gpu-device-id",
        "gpu-use-dp",
        "buy-min",
        "buy-max",
        "buy-step",
        "sell-min",
        "sell-max",
        "sell-step",
        "fee-bps",
        "slippage-bps",
        "min-trades",
        "min-trades-per-day",
        "min-sharpe",
        "max-drawdown-limit",
        "min-return",
        "optimize",
        "position-mode",
    ),
    "backtest": (
        "test-ratio",
        "buy-threshold",
        "sell-threshold",
        "position-mode",
        "fee-bps",
        "slippage-bps",
    ),
    "backtest-best": (
        "test-ratio",
        "buy-threshold",
        "sell-threshold",
        "position-mode",
        "fee-bps",
        "slippage-bps",
    ),
    "grid": (
        "test-ratio",
        "fee-bps",
        "slippage-bps",
        "buy-min",
        "buy-max",
        "buy-step",
        "sell-min",
        "sell-max",
        "sell-step",
        "min-trades",
        "optimize",
        "top-k",
    ),
}


def _common_config(config: dict) -> dict:
    value = config.get("common", {})
    return value if isinstance(value, dict) else {}


def _command_config(config: dict, command: str) -> dict:
    commands = config.get("commands", {})
    if not isinstance(commands, dict):
        return {}
    value = commands.get(command, {})
    return value if isinstance(value, dict) else {}


def _get_config_value(config: dict, command: str, key: str, default=None):
    cmd_cfg = _command_config(config, command)
    if key in cmd_cfg:
        return cmd_cfg[key]
    common_cfg = _common_config(config)
    if key in common_cfg:
        return common_cfg[key]
    if key == "symbol" and "symbol" in config:
        return config["symbol"]
    return default


def _value_to_cli_args(key: str, value) -> list[str]:
    if value is None:
        return []
    flag = f"--{key}"
    if isinstance(value, bool):
        return [flag] if value else []
    if isinstance(value, list):
        return [flag, ",".join(str(x) for x in value)]
    return [flag, str(value)]


def _command_args_from_config(config: dict, command: str, excludes: set[str] | None = None) -> list[str]:
    excludes = excludes or set()
    args: list[str] = []
    cmd_cfg = _command_config(config, command)
    for key, value in cmd_cfg.items():
        if key in excludes:
            continue
        args.extend(_value_to_cli_args(key, value))
    return args


def _resolve_symbol(config: dict, command: str, cli_symbol: str | None) -> str:
    if cli_symbol:
        return cli_symbol
    from_config = _get_config_value(config, command, "symbol", None)
    if from_config:
        return str(from_config)
    print(f"error: SYMBOL is required for '{command}'", file=sys.stderr)
    print(f"example: python run_ml.py {command} ETHUSDT", file=sys.stderr)
    print("or set symbol in run_ml.config.json", file=sys.stderr)
    raise SystemExit(2)


def _validate_required_command_keys(config: dict, command: str) -> None:
    required = REQUIRED_COMMAND_KEYS.get(command)
    if not required:
        return
    cmd_cfg = _command_config(config, command)
    missing = [k for k in required if k not in cmd_cfg]
    if not missing:
        return
    miss = ", ".join(missing)
    raise SystemExit(
        f"missing required config keys for '{command}': {miss}\n"
        f"please fill commands.{command} in run_ml.config.json (see run_ml.config.example.json)"
    )


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


def _render_progress(prefix: str, index: int, total: int, status: str, label: str) -> None:
    percent = int(index * 100 / total) if total > 0 else 100
    line = f"\r[setup] {prefix} {index}/{total} ({percent}%) {status:<10} {label}"
    end = "\n" if index >= total else ""
    print(line, end=end, flush=True)


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

    download_plan: list[tuple[str, Path, str]] = []
    if time_period == "daily":
        cur = start_day
        while cur <= end_day:
            day_str = cur.isoformat()
            file_name = f"{symbol_upper}-{interval}-{day_str}.zip"
            url = f"{BINANCE_DATA_BASE_URL}/{trading_type}/daily/klines/{symbol_upper}/{interval}/{file_name}"
            download_plan.append((url, save_dir / file_name, day_str))
            cur += timedelta(days=1)
    elif time_period == "monthly":
        for month_start in _iter_month_starts(start_day, end_day):
            month_str = f"{month_start.year:04d}-{month_start.month:02d}"
            file_name = f"{symbol_upper}-{interval}-{month_str}.zip"
            url = f"{BINANCE_DATA_BASE_URL}/{trading_type}/monthly/klines/{symbol_upper}/{interval}/{file_name}"
            download_plan.append((url, save_dir / file_name, month_str))
    else:
        raise SystemExit(f"unsupported time-period: {time_period}, expected daily or monthly")

    total = len(download_plan)
    if total == 0:
        raise SystemExit("download plan is empty, please check start-date/end-date")

    downloaded = 0
    exists = 0
    not_found = 0
    print(
        f"[setup] data not found, downloading klines from Binance: symbol={symbol_upper} interval={interval} type={trading_type} period={time_period}",
        flush=True,
    )
    print(f"[setup] download plan: {total} files", flush=True)
    for i, (url, target_path, label) in enumerate(download_plan, start=1):
        status = _download_to_file(url, target_path)
        if status == "downloaded":
            downloaded += 1
        elif status == "exists":
            exists += 1
        else:
            not_found += 1
        _render_progress("download", i, total, status, label)

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
  python run_ml.py <command> [SYMBOL] [--config path.json]

Commands:
  data          Download kline data only (no training)
  train         Train one model
  auto          Auto-search long_only config
  auto-ls       Auto-search long_short config
  backtest      Backtest quant model
  backtest-best Backtest best long_short model
  grid          Grid-search thresholds on quant model

Config:
  --config <path>    Load settings from JSON config.
  default config     If ./run_ml.config.json exists, it is auto-loaded.
  precedence         command config > common config; SYMBOL can be overridden by CLI.

Examples:
  python run_ml.py train --config run_ml.config.json
  python run_ml.py data ETHUSDT
  python run_ml.py train ETHUSDT
  python run_ml.py auto-ls ETHUSDT
  python run_ml.py backtest ETHUSDT
"""


@dataclass(frozen=True)
class CommonSettings:
    interval: str
    trading_type: str
    time_period: str
    start_date: str
    end_date: str


@dataclass(frozen=True)
class CommandContext:
    command: str
    symbol: str
    prefix: str
    settings: CommonSettings


def _as_config_str(value, default: str) -> str:
    if value is None:
        return default
    return str(value)


def _resolve_common_settings(config: dict, command: str) -> CommonSettings:
    return CommonSettings(
        interval=_as_config_str(_get_config_value(config, command, "interval", INTERVAL), INTERVAL),
        trading_type=_as_config_str(_get_config_value(config, command, "trading-type", TRADING_TYPE), TRADING_TYPE),
        time_period=_as_config_str(_get_config_value(config, command, "time-period", TIME_PERIOD), TIME_PERIOD),
        start_date=_as_config_str(_get_config_value(config, command, "start-date", START_DATE), START_DATE),
        end_date=_as_config_str(_get_config_value(config, command, "end-date", END_DATE), END_DATE),
    )


def _pairs_to_cli_args(pairs: list[tuple[str, object]]) -> list[str]:
    args: list[str] = []
    for key, value in pairs:
        args.extend(_value_to_cli_args(key, value))
    return args


def _base_script_pairs(ctx: CommandContext) -> list[tuple[str, object]]:
    pairs: list[tuple[str, object]] = [
        ("data-root", str(DATA_ROOT)),
        ("symbol", ctx.symbol),
        ("interval", ctx.settings.interval),
        ("trading-type", ctx.settings.trading_type),
        ("time-period", ctx.settings.time_period),
        ("start-date", ctx.settings.start_date),
    ]
    if ctx.settings.end_date:
        pairs.append(("end-date", ctx.settings.end_date))
    return pairs


def _prepare_context(symbol: str, config: dict, command: str) -> CommandContext:
    ensure_runtime()
    settings = _resolve_common_settings(config, command)
    ensure_kline_data(
        symbol=symbol,
        interval=settings.interval,
        trading_type=settings.trading_type,
        time_period=settings.time_period,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    return CommandContext(command=command, symbol=symbol, prefix=to_prefix(symbol), settings=settings)


def _run_script_command(
    ctx: CommandContext,
    config: dict,
    script_path: str,
    default_pairs: list[tuple[str, object]],
    config_command: str | None = None,
    config_excludes: set[str] | None = None,
) -> None:
    excludes = set(COMMON_CONFIG_KEYS)
    if config_excludes:
        excludes.update(config_excludes)
    cmd = [str(VENV_PY), script_path]
    cmd.extend(_pairs_to_cli_args(_base_script_pairs(ctx) + default_pairs))
    cmd.extend(_command_args_from_config(config, config_command or ctx.command, excludes))
    run(cmd)


def _resolve_best_model(prefix: str, symbol: str) -> Path:
    best_model_refine = Path(f"./models/{prefix}_best_ls_refine_model.txt")
    best_model = best_model_refine if best_model_refine.is_file() else Path(f"./models/{prefix}_best_ls_model.txt")
    if best_model.is_file():
        return best_model
    raise SystemExit(
        f"best model not found: ./models/{prefix}_best_ls_refine_model.txt or ./models/{prefix}_best_ls_model.txt\n"
        f"run auto-ls first: python run_ml.py auto-ls {symbol}"
    )


def _load_best_thresholds(prefix: str) -> tuple[float, float, Path] | None:
    best_cfg = Path(f"./models/{prefix}_best_ls_config.json")
    if not best_cfg.is_file():
        return None

    try:
        payload = json.loads(best_cfg.read_text(encoding="utf-8"))
        best = payload.get("best")
        if not isinstance(best, dict):
            raise ValueError("missing 'best' object")
        buy_th = float(best["buy_threshold"])
        sell_th = float(best["sell_threshold"])
        if not (0.0 <= sell_th < buy_th <= 1.0):
            raise ValueError(f"invalid thresholds: buy={buy_th}, sell={sell_th}")
    except Exception as exc:
        print(f"[WARN] ignore best config thresholds ({best_cfg}): {exc}", flush=True)
        return None
    return buy_th, sell_th, best_cfg


def run_train(symbol: str, config: dict) -> None:
    _validate_required_command_keys(config, "train")
    ctx = _prepare_context(symbol, config, "train")
    print(
        f"[train] symbol={symbol} interval={ctx.settings.interval} start_date={ctx.settings.start_date} device=cuda out_prefix={ctx.prefix}",
        flush=True,
    )
    _run_script_command(
        ctx=ctx,
        config=config,
        script_path="scripts/train_lightgbm.py",
        default_pairs=[
            ("model-out", f"./models/{ctx.prefix}_quant_model.txt"),
            ("metrics-out", f"./models/{ctx.prefix}_metrics.json"),
            ("importance-out", f"./models/{ctx.prefix}_feature_importance.csv"),
        ],
    )


def run_data(symbol: str, config: dict) -> None:
    ctx = _prepare_context(symbol, config, "data")
    print(
        f"[data] symbol={symbol} interval={ctx.settings.interval} type={ctx.settings.trading_type} "
        f"period={ctx.settings.time_period} start={ctx.settings.start_date} end={ctx.settings.end_date or 'latest'}",
        flush=True,
    )
    save_dir = (
        DATA_ROOT
        / "futures"
        / ctx.settings.trading_type
        / ctx.settings.time_period
        / "klines"
        / symbol.upper()
        / ctx.settings.interval
    )
    print(f"[data] ready: {save_dir}", flush=True)


def run_auto(symbol: str, config: dict, long_short: bool) -> None:
    command = "auto-ls" if long_short else "auto"
    _validate_required_command_keys(config, command)
    ctx = _prepare_context(symbol, config, command)
    mode = str(_command_config(config, command).get("position-mode"))
    print(
        f"[auto] symbol={symbol} mode={mode} interval={ctx.settings.interval} start_date={ctx.settings.start_date} device=cuda",
        flush=True,
    )
    suffix = "_ls" if long_short else ""
    default_pairs: list[tuple[str, object]] = [
        ("results-out", f"./models/{ctx.prefix}_auto_search{suffix}_results.csv"),
        ("best-json-out", f"./models/{ctx.prefix}_best{suffix}_config.json"),
        ("best-model-out", f"./models/{ctx.prefix}_best{suffix}_model.txt"),
        ("best-equity-out", f"./models/{ctx.prefix}_best{suffix}_equity.csv"),
    ]

    _run_script_command(
        ctx=ctx,
        config=config,
        script_path="scripts/auto_search_eth.py",
        default_pairs=default_pairs,
        config_command=command,
    )


def run_backtest(symbol: str, config: dict, best: bool) -> None:
    command = "backtest-best" if best else "backtest"
    _validate_required_command_keys(config, command)
    ctx = _prepare_context(symbol, config, command)

    if best:
        model_path = _resolve_best_model(ctx.prefix, symbol)
        mode = str(_command_config(config, command).get("position-mode"))
        print(f"[backtest] symbol={symbol} model={model_path} mode={mode}", flush=True)
        buy_th = float(_get_config_value(config, command, "buy-threshold"))
        sell_th = float(_get_config_value(config, command, "sell-threshold"))
        threshold_source = "run_ml.config.json"
        best_thresholds = _load_best_thresholds(ctx.prefix)
        if best_thresholds is not None:
            buy_th, sell_th, best_cfg = best_thresholds
            threshold_source = str(best_cfg)
        print(
            f"[backtest] thresholds buy={buy_th:.6f} sell={sell_th:.6f} source={threshold_source}",
            flush=True,
        )
        default_pairs: list[tuple[str, object]] = [
            ("model-path", str(model_path)),
            ("buy-threshold", buy_th),
            ("sell-threshold", sell_th),
            ("summary-out", f"./models/{ctx.prefix}_best_live_summary.json"),
            ("equity-out", f"./models/{ctx.prefix}_best_live_equity.csv"),
        ]
    else:
        print(f"[backtest] symbol={symbol} model=./models/{ctx.prefix}_quant_model.txt", flush=True)
        default_pairs = [
            ("model-path", f"./models/{ctx.prefix}_quant_model.txt"),
            ("summary-out", f"./models/{ctx.prefix}_backtest_summary.json"),
            ("equity-out", f"./models/{ctx.prefix}_backtest_equity.csv"),
        ]

    _run_script_command(
        ctx=ctx,
        config=config,
        script_path="scripts/backtest_lightgbm.py",
        default_pairs=default_pairs,
        config_command=command,
        config_excludes={"buy-threshold", "sell-threshold"} if best else None,
    )


def run_grid(symbol: str, config: dict) -> None:
    _validate_required_command_keys(config, "grid")
    ctx = _prepare_context(symbol, config, "grid")
    print(f"[grid] symbol={symbol} model=./models/{ctx.prefix}_quant_model.txt", flush=True)
    _run_script_command(
        ctx=ctx,
        config=config,
        script_path="scripts/grid_search_backtest.py",
        default_pairs=[
            ("model-path", f"./models/{ctx.prefix}_quant_model.txt"),
            ("results-out", f"./models/{ctx.prefix}_grid_results.csv"),
            ("best-out", f"./models/{ctx.prefix}_grid_best.json"),
            ("best-equity-out", f"./models/{ctx.prefix}_grid_best_equity.csv"),
        ],
    )


def run_auto_ls(symbol: str, config: dict) -> None:
    run_auto(symbol, config, long_short=True)


def run_auto_long_only(symbol: str, config: dict) -> None:
    run_auto(symbol, config, long_short=False)


def run_backtest_best(symbol: str, config: dict) -> None:
    run_backtest(symbol, config, best=True)


def run_backtest_quant(symbol: str, config: dict) -> None:
    run_backtest(symbol, config, best=False)


COMMAND_RUNNERS = {
    "data": run_data,
    "train": run_train,
    "auto": run_auto_long_only,
    "auto-ls": run_auto_ls,
    "backtest": run_backtest_quant,
    "backtest-best": run_backtest_best,
    "grid": run_grid,
}


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(usage())
        return 0

    ensure_supported_platform()

    cmd = argv[0]
    args = argv[1:]
    if cmd not in SUPPORTED_COMMANDS:
        print(f"unknown command: {cmd}", file=sys.stderr)
        print()
        print(usage())
        return 1

    config_path, args_wo_config = _extract_config_arg(args)
    config = _load_config(config_path)
    cli_symbol: str | None = None
    if args_wo_config and not args_wo_config[0].startswith("-"):
        cli_symbol = args_wo_config[0]
        args_wo_config = args_wo_config[1:]
    if args_wo_config:
        raise SystemExit(
            "command-line parameter overrides are disabled; put parameters in run_ml.config.json or pass --config path"
        )
    symbol = _resolve_symbol(config, cmd, cli_symbol)

    COMMAND_RUNNERS[cmd](symbol, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
