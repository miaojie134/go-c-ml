# LightGBM Quant Starter

基于 Binance 官方 `binance-public-data` 下载的 Kline zip 数据，做特征工程并训练一个二分类 LightGBM 模型：

- 标签：未来 `N` 根 K 线收益率是否大于阈值
- 输出：
  - 模型文件：`models/eth_quant_model.txt`
  - 评估指标：`models/eth_metrics.json`
  - 特征重要性：`models/eth_feature_importance.csv`

## 1) 安装依赖

```bash
python -m pip install -U pip
python -m pip install -r requirements.txt
```

## 2) 训练（ETH 15m）

```bash
./run_eth.sh
```

或手动命令：

```bash
python train_lightgbm.py \
  --data-root ./binance-public-data/python/data \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --horizon 2 \
  --target-threshold 0.0
```

## 3) 训练（SOL 15m）

```bash
python train_lightgbm.py \
  --data-root ./binance-public-data/python/data \
  --symbol SOLUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2022-01-01 \
  --horizon 2 \
  --target-threshold 0.0 \
  --model-out models/sol_quant_model.txt \
  --metrics-out models/sol_metrics.json \
  --importance-out models/sol_feature_importance.csv
```

## 4) 回测（ETH，一条命令）

```bash
./run_backtest_eth.sh
```

输出：

- `models/eth_backtest_summary.json`
- `models/eth_backtest_equity.csv`

## 5) 自动搜索最优模型（推荐）

先跑多空版自动搜索：

```bash
./run_auto_search_eth_ls.sh
```

再跑精细搜索（更高收益）：

```bash
./binance-public-data/python/.venv/bin/python auto_search_eth.py \
  --data-root ./binance-public-data/python/data \
  --symbol ETHUSDT \
  --interval 15m \
  --trading-type um \
  --time-period daily \
  --start-date 2023-01-01 \
  --train-ratio 0.7 \
  --val-ratio 0.85 \
  --horizons 3,4,5 \
  --target-thresholds 0.0003,0.0005,0.0007,0.001 \
  --learning-rates 0.03,0.05,0.07 \
  --num-leaves 15,31,63,127 \
  --n-estimators 900 \
  --early-stop-rounds 100 \
  --buy-min 0.50 \
  --buy-max 0.64 \
  --buy-step 0.01 \
  --sell-min 0.14 \
  --sell-max 0.34 \
  --sell-step 0.01 \
  --fee-bps 5 \
  --slippage-bps 1 \
  --min-trades 20 \
  --optimize strategy_total_return \
  --position-mode long_short \
  --results-out ./models/eth_auto_search_ls_refine_results.csv \
  --best-json-out ./models/eth_best_ls_refine_config.json \
  --best-model-out ./models/eth_best_ls_refine_model.txt \
  --best-equity-out ./models/eth_best_ls_refine_equity.csv
```

使用最优模型一键回测：

```bash
./run_backtest_eth_best.sh
```

核心输出：

- `models/eth_best_ls_refine_model.txt`
- `models/eth_best_ls_refine_config.json`
- `models/eth_best_live_summary.json`
- `models/eth_best_live_signal_config.json`（含特征顺序和执行阈值，供 Go 推理对接）

## GPU 训练

如果你在有显卡机器上跑，可在训练命令后加：

```bash
--device-type gpu --gpu-platform-id 0 --gpu-device-id 0 --gpu-use-dp 0
```

如果你是 NVIDIA + CUDA 版 LightGBM，可把 `--device-type gpu` 改成 `--device-type cuda`。
脚本会在 GPU 不可用时自动回退到 CPU，并打印 warning。

## 常用参数

- `--horizon 2`：预测未来 2 根 K 线
- `--target-threshold 0.001`：预测未来收益 > 0.1%
- `--train-ratio 0.8`：前 80% 时间段训练，后 20% 测试（避免未来数据泄露）
