# LightGBM Quant Starter

基于 Binance 官方 `binance-public-data` 下载的 Kline zip 数据，做特征工程并训练一个二分类 LightGBM 模型：

- 标签：未来 `N` 根 K 线收益率是否大于阈值
- 输出：
  - 模型文件：`models/eth_quant_model.txt`
  - 评估指标：`models/eth_metrics.json`
  - 特征重要性：`models/eth_feature_importance.csv`

核心代码文件已整理到 `scripts/` 目录。

## 1) 安装依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2) 一条命令开始训练（推荐）

```bash
# Linux / macOS
./run_ml.sh train ETHUSDT

# Windows (CMD / PowerShell)
run_ml.cmd train ETHUSDT
# 或
py run_ml.py train ETHUSDT
```

输出到：

- `models/eth_quant_model.txt`
- `models/eth_metrics.json`
- `models/eth_feature_importance.csv`

## 3) 统一入口命令

```bash
# Linux / macOS
./run_ml.sh train SYMBOL [extra args...]
./run_ml.sh auto SYMBOL [extra args...]
./run_ml.sh auto-ls SYMBOL [extra args...]
./run_ml.sh backtest SYMBOL [extra args...]
./run_ml.sh backtest-best SYMBOL [extra args...]
./run_ml.sh grid SYMBOL [extra args...]

# Windows
run_ml.cmd train SYMBOL [extra args...]
run_ml.cmd auto SYMBOL [extra args...]
run_ml.cmd auto-ls SYMBOL [extra args...]
run_ml.cmd backtest SYMBOL [extra args...]
run_ml.cmd backtest-best SYMBOL [extra args...]
run_ml.cmd grid SYMBOL [extra args...]
```

例子：

```bash
./run_ml.sh train BTCUSDT
./run_ml.sh train SOLUSDT --start-date 2022-01-01 --horizon 3
./run_ml.sh auto-ls ETHUSDT
./run_ml.sh backtest ETHUSDT
./run_ml.sh backtest-best ETHUSDT
./run_ml.sh grid ETHUSDT
```

## 4) 自动准备环境和数据

首次运行 `run_ml.sh` 会自动执行：

- 缺少 `binance-public-data` 时自动 `git clone`
- 缺少虚拟环境时自动创建 `venv`
- 缺少依赖时自动 `pip install -r requirements.txt`
- 若交易对 K 线数据不存在，自动从 Binance 下载对应 `symbol/interval` 数据

`run_ml.cmd` / `run_ml.py` 在 Windows 下也有同样行为。

如果你传了自定义 `DATA_ROOT`，请确保它以 `/data` 结尾，便于自动下载器写入正确目录。

## 5) 输出文件命名

输出文件会用交易对前缀命名（如 `ETHUSDT -> eth_*`, `BTCUSDT -> btc_*`）：

- 训练：`models/<prefix>_quant_model.txt` / `models/<prefix>_metrics.json`
- 自动搜索：`models/<prefix>_auto_search*.csv`、`models/<prefix>_best*.json/txt`
- 回测：`models/<prefix>_backtest_*.json/csv`、`models/<prefix>_best_live_*.json/csv`

## 6) 手动运行底层命令（可选）

```bash
python scripts/train_lightgbm.py \
  --data-root ./binance-public-data/python/data \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --horizon 2 \
  --target-threshold 0.0
```

## 7) GPU 训练

当前默认 `--device-type` 已是 `cuda`。如果你要手动覆盖，可在训练命令后加：

```bash
--device-type gpu --gpu-platform-id 0 --gpu-device-id 0 --gpu-use-dp 0
```

如果你是 NVIDIA + CUDA 版 LightGBM，可把 `--device-type gpu` 改成 `--device-type cuda`。
脚本会在 GPU 不可用时自动回退到 CPU，并打印 warning。

## 8) 常用参数

- `--horizon 2`：预测未来 2 根 K 线
- `--target-threshold 0.001`：预测未来收益 > 0.1%
- `--train-ratio 0.8`：前 80% 时间段训练，后 20% 测试（避免未来数据泄露）
