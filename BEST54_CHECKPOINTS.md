# 保存54组最佳配置模型，后续独立评估

这一步只重训和保存模型，不输出用户/物品四组指标。

## 配置来源

`best54_configs.json` 已冻结九个模型、三个数据集、Original/Item-only 两个版本的54组参数。与2026-10-10的29B LaTeX表相同：Original 从第一轮参数网格选择，Item-only 从完整29种B的联合网格选择，按每个版本的 test Recall@20 选配置。Amazon/MixGCF 的 Item-only 使用第一轮 sym。

此JSON位于仓库根目录，可以正常推送。服务器不需要被Git忽略的 `analysis/`、`logs/` 或本地Excel。JSON保留历史最佳指标和日志相对路径用于追溯，运行时不读取历史日志。

每组只用一个seed=2025。重训仍按原逻辑挑选最佳epoch：Ali/Amazon用validation Recall@20，Yelp2018用test Recall@20，10轮无提升早停。重训结果可能因设备/软件和浮点计算差异而与历史记录不同，新的result和checkpoint记录实际结果。不会为了追逐历史测试分数反复重训或替换其他参数。

## 六张卡启动

在服务器ItemGCN目录，激活原来的训练环境后：

```bash
# 仅检查54组任务、数据指纹和启动参数，不训练、不创建任务目录
bash start_best54_checkpoints.sh 0 1 2 3 4 5 --dry-run

# 正式启动，脚本内部已经使用nohup
bash start_best54_checkpoints.sh 0 1 2 3 4 5
```

| GPU位置 | 任务 | 训练次数 |
|---|---|---:|
| 第1个卡号 | Ali / Original | 9 |
| 第2个卡号 | Ali / Item-only | 9 |
| 第3个卡号 | Amazon / Original | 9 |
| 第4个卡号 | Amazon / Item-only | 9 |
| 第5个卡号 | Yelp2018 / Original | 9 |
| 第6个卡号 | Yelp2018 / Item-only | 9 |

每张卡顺序跑9个模型，不是在一张卡同时启动9次训练。改卡号即可换GPU，例如 `2 3 4 5 6 7`。脚本提交后会打印PID和日志路径；提交不等于训练完成。

默认输出根目录 `checkpoints/best54_20261010/` 已被 `.gitignore` 排除。用 `--output-root checkpoints/自定义目录` 更改目录，`--data-path /path/to/data` 指定数据根目录。可设置 `PYTHON=/path/to/python` 指定解释器。

只用一张卡依次运行全部54组：

```bash
nohup python -u run_best_checkpoints.py --gpu-id 0 > best54.out 2>&1 < /dev/null &
```

也可以每个数据集一张卡，每张跑18组：

```bash
nohup python -u run_best_checkpoints.py --dataset ali --gpu-id 0 > best54_ali.out 2>&1 < /dev/null &
nohup python -u run_best_checkpoints.py --dataset amazon --gpu-id 1 > best54_amazon.out 2>&1 < /dev/null &
nohup python -u run_best_checkpoints.py --dataset yelp2018 --gpu-id 2 > best54_yelp.out 2>&1 < /dev/null &
```

以上是替代启动方式，选择其中一种即可，不要对相同输出同时启动多套调度器。Python入口还支持 `--phase original/item_only` 和 `--num-shards N --shard-index I`；分片编号从0开始，所有分片使用相同筛选条件可覆盖所选任务。

## 保存内容

例如 `checkpoints/best54_20261010/ali/rns/item_only/attempt_001/`：

- `model_.ckpt`：最佳epoch的完整模型state_dict，包括持久化buffer，CPU格式保存，可在另一台机器重新加载。
- `checkpoint.json`：对应epoch及全部指标、模型参数、用户/物品数量、数据文件SHA256、权重SHA256、代码摘要和运行环境。
- `config.json`、`environment.json`：这次实际训练设置和环境。
- `result.json`、`metrics.csv`、`train.log`：最终结果、逐epoch指标和完整日志。
- `command.json`：实际训练命令。

最佳权重在指标严格改善时覆盖，训练结束后的最后一轮不会覆盖更好的历史epoch。保存仅用于推理/评估，不包含优化器状态，不提供从中断epoch继续训练。

父目录 `status.json` 指向当前attempt，`manifest.json` 冻结配置和训练代码。输出根目录的 `index_*.json` 记录各调度器任务状态、checkpoint相对路径、新结果和历史参考Recall@20。

重复运行同一启动命令会跳过有完整权重、匹配元数据且成功结束的任务。只有旧成功日志但丢失/损坏权重时会创建新的attempt重训；旧日志不会被覆盖。突然关机留下的 `.running.lock` 必须先确认没有旧进程，再手动处理。代码或配置改变时应使用新的输出根目录。

## 后续如何加载

取回整个任务输出目录，同时保留本版本代码和原始 `data/<dataset>/train.txt`、`valid.txt`（存在时）、`test.txt`。图结构不属于可学习权重，加载时会用原训练交互重建，因此不能只有一个ckpt而丢失数据。加载器会校验数据指纹和用户/物品ID空间，防止换了划分却继续使用旧权重。

先对某个模型做普通整体测试，无需训练：

```bash
python evaluate_checkpoint.py \
  --checkpoint checkpoints/best54_20261010/ali/rns/item_only/attempt_001/model_.ckpt \
  --data-path data --gpu-id 0 --ks 10 20 50 \
  --output experiment_results/checkpoint_eval/ali_rns_item_only.json
```

也可直接传attempt目录。重试后以父目录status.json中的attempt为准，不要总是假设attempt_001。现有输出不会被覆盖。CPU评估用 `--cpu`。该入口只做整体评估，不做四组分析。

后续编写分组脚本时复用：

```python
from utils.checkpoint import load_for_evaluation

model, args, data, metadata = load_for_evaluation(
    checkpoint_path, data_path='data', cuda=False)
# data与load_data返回顺序一致，可取得训练历史、测试正例、映射后的ID。
# model已切换eval模式；后续推理使用torch.no_grad()。
```

用户/物品分组、不同K和其他评估统计都可以在这些保存好的模型上继续做。注意现有evaluate.py为了加速命中判断会临时置换分数列；以后按物品分组时需保留或还原真实item ID，不应把其内部topk位置直接当作物品ID。
