# go-c-ml

配置驱动的 LightGBM 量化训练框架。

核心原则：
- 所有训练/回测调参都写在 `run_ml.config.json`
- 命令行只保留：`python run_ml.py <command> [SYMBOL] [--config path]`
- 默认 GPU：`device-type=cuda`

## 快速开始（推荐流程）

1) 准备配置文件：

```bash
cp run_ml.config.example.json run_ml.config.json
```

2) 补齐数据：

```bash
python run_ml.py data
```

3) 自动搜索多空最优参数并训练：

```bash
python run_ml.py auto-ls
```

4) 用最优模型回测：

```bash
python run_ml.py backtest-best
```

5) 导出给 Go/其他项目：

```bash
python run_ml.py bundle
```

## 命令说明（精简版）

主流程常用：
- `data`：下载/补齐指定时间范围的数据
- `auto-ls`：多空自动调参 + 训练（主命令）
- `backtest-best`：回测 `auto-ls` 产出的最优模型
- `bundle`：导出可部署 JSON（含特征列表、阈值、模型路径）

可选命令：
- `train`：训练单一基线模型（用于排障/对照，不是主流程必需）
- `auto`：只做多自动调参
- `backtest`：回测 `train` 训练出的基线模型
- `grid`：固定模型后做阈值网格搜索

说明：
- `SYMBOL` 可省略（从配置读），也可临时覆盖，如 `python run_ml.py auto-ls BTCUSDT`
- 不支持命令行传其它参数（例如 `--start-date`），避免配置漂移

## 配置文件

默认读取项目根目录：`run_ml.config.json`

优先级：
- `commands.<command>` > `common`
- `SYMBOL`：命令行 > 配置 `symbol`

重点配置：
- `common.*`：`interval`, `trading-type`, `time-period`, `start-date`, `end-date`
- `commands.auto-ls.*`：模型超参 + 阈值搜索区间 + 交易约束
- `commands.backtest-best.*`：回测参数
- `commands.bundle.*`：导出参数（`mode`, `out`）

## 输出文件

以 `ETHUSDT` 为例：
- 自动搜索结果：`models/eth_auto_search_ls_results.csv`
- 最优配置：`models/eth_best_ls_config.json`
- 最优模型：`models/eth_best_ls_model.txt`
- 回测结果：`models/eth_best_live_summary.json`, `models/eth_best_live_equity.csv`
- 部署包：`models/eth_best_ls_bundle.json`

## 给 Go 项目使用

`bundle` 文件里包含：
- 模型路径（LightGBM txt）
- `inference.feature_names`（推理时必须按该顺序构造特征向量）
- `strategy`（`position_mode`, `horizon`, `target_threshold`, `buy_threshold`, `sell_threshold`）

最小集成流程：
1. 读取 `*_bundle.json`
2. 按 `feature_names` 顺序生成特征数组
3. LightGBM 预测得到 `proba_up`
4. 按阈值执行交易规则

## 常见问题

`train` 命令有必要吗？
- 对主流程不是必需。
- 你要自动调参时，直接用 `auto-ls`。

为什么有数据却没补下载？
- 新版本会按请求时间范围检查缺失文件并补齐，不再“检测到任意文件就跳过”。
