# LightGBM Quant Starter

基于 Binance 官方公开 Kline 数据，做特征工程并训练一个二分类 LightGBM 模型：

- 标签：未来 `N` 根 K 线收益率是否大于阈值
- 输出：
  - 模型文件：`models/eth_quant_model.txt`
  - 评估指标：`models/eth_metrics.json`
  - 特征重要性：`models/eth_feature_importance.csv`

核心代码文件已整理到 `scripts/` 目录。

当前脚本结构：

- `scripts/ml_common.py`：公共能力（数据加载、特征工程、回测仿真、统计指标）
- `scripts/train_lightgbm.py`：训练
- `scripts/auto_search_eth.py`：自动搜索模型参数 + 阈值
- `scripts/backtest_lightgbm.py`：单模型回测
- `scripts/grid_search_backtest.py`：阈值网格回测

## 1) 安装依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2) 一条命令开始训练（推荐）

先准备配置文件（一次）：

```bash
# Linux / macOS
cp run_ml.config.example.json run_ml.config.json

# Windows
copy run_ml.config.example.json run_ml.config.json
```

然后直接跑（自动读取 `run_ml.config.json`）：

```bash
python run_ml.py train
```

输出到：

- `models/eth_quant_model.txt`
- `models/eth_metrics.json`
- `models/eth_feature_importance.csv`

## 3) 统一入口命令

```bash
python run_ml.py data [SYMBOL]
python run_ml.py train [SYMBOL]
python run_ml.py auto [SYMBOL]
python run_ml.py auto-ls [SYMBOL]
python run_ml.py backtest [SYMBOL]
python run_ml.py backtest-best [SYMBOL]
python run_ml.py grid [SYMBOL]
```

说明：

- 参数统一从配置文件读取
- `SYMBOL` 可省略（从配置读取），也可在命令行临时覆盖
- 命令行不再支持其它参数覆盖（如 `--start-date`），避免混乱
- 调参与回测参数请全部写入 `run_ml.config.json` 的 `commands.<command>` 节点（不再建议写死在代码里）
- 对 `train/auto/auto-ls/backtest/backtest-best/grid`：关键调参项缺失会直接报错，避免偷偷使用脚本默认值

指定配置文件：

```bash
python run_ml.py train --config run_ml.config.json
python run_ml.py auto-ls --config run_ml.config.json
```

例子：

```bash
python run_ml.py data
python run_ml.py train
python run_ml.py auto-ls
python run_ml.py backtest
```

## 4) 自动准备环境和数据

首次运行 `run_ml.py` 会自动执行：

- 优先复用项目根目录 `.venv`；若不存在才自动创建
- 缺少依赖时自动 `pip install -r requirements.txt`
- 若交易对 K 线数据不存在，自动从 Binance 下载对应 `symbol/interval` 数据到 `./data`
- 下载会显示进度：总文件数、当前文件、百分比、下载状态

Windows 建议在已激活虚拟环境后使用 `python run_ml.py ...`，不要用 `py run_ml.py ...`（`py` 可能跳过当前 venv）。

如果你传了自定义 `DATA_ROOT`，脚本会把数据下载到该目录下的标准结构中。

常用配置项（写在 `run_ml.config.json`）：

- `common.interval`：K 线周期（如 `1m` / `5m` / `15m` / `1h`）
- `common.time-period`：下载粒度（`daily` 或 `monthly`）
- `common.start-date`：起始日期（`YYYY-MM-DD`）
- `common.end-date`：结束日期（`YYYY-MM-DD`，不传则到最新）
- `commands.train.*`：训练参数（horizon、target-threshold、train-ratio、device-type 等）
- `commands.auto.*` / `commands.auto-ls.*`：自动搜索参数（网格、阈值、交易约束、模型超参）
- `commands.backtest.*` / `commands.backtest-best.*`：回测参数（threshold、fee、slippage、position-mode、test-ratio）
- `commands.grid.*`：网格回测参数（buy/sell 搜索区间、步长、optimize、top-k）

配置文件说明：

- 默认会自动读取项目根目录 `run_ml.config.json`
- 也可通过 `--config path.json` 指定
- 优先级：`commands.<cmd>` > `common`
- `SYMBOL` 优先级：命令行 `SYMBOL` > 配置中的 `symbol`
- 完整模板见：`run_ml.config.example.json`
- 建议先完整复制模板，再按交易对和策略需求调参

## 5) 输出文件命名

输出文件会用交易对前缀命名（如 `ETHUSDT -> eth_*`, `BTCUSDT -> btc_*`）：

- 训练：`models/<prefix>_quant_model.txt` / `models/<prefix>_metrics.json`
- 自动搜索：`models/<prefix>_auto_search*.csv`、`models/<prefix>_best*.json/txt`
- 回测：`models/<prefix>_backtest_*.json/csv`、`models/<prefix>_best_live_*.json/csv`

## 6) 手动运行底层命令（可选）

```bash
python scripts/train_lightgbm.py \
  --data-root ./data \
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
