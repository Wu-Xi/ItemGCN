# 模型参数 × 用户表征构造：补齐联合搜索（默认排除 sym）

入口 `run_joint_user_representation.py` 读取 `experiments.json` 的全部模型参数组合，按“模型 → 模型参数组合 → B 构造”的顺序执行。每一组模型参数都训练除 sym 外的 28 种 B，不再只使用模型网格的第一组。第一轮已在每组模型参数上运行 sym，本轮默认不重复。只运行 Item-only；Original 不依赖 B。需要完整同环境重跑时可添加 --include-sym，恢复 29 种。

## 推荐：四组模型、十二张 GPU 并行

每个分组启动文件依次完成三个数据集的参数预览验证，然后启动三个后台调度进程。每个进程占用指定的一张 GPU、负责一个数据集；同组多模型在该进程中按顺序运行。

| 启动文件 | 模型 | Ali GPU | Amazon GPU | Yelp2018 GPU | 每数据集任务数 |
|---|---|---:|---:|---:|---:|
| `start_joint_recdcl.sh` | RecDCL | 0 | 1 | 2 | 81 × 28 = 2268 |
| `start_joint_sgl.sh` | SGL | 3 | 4 | 5 | 24 × 28 = 672 |
| `start_joint_stabcf_ahns.sh` | StabCF、AHNS | 6 | 7 | 8 | (12 + 12) × 28 = 672 |
| `start_joint_remaining.sh` | RNS、MixGCF、DirectAU、GraphAU、SimGCL | 9 | 10 | 11 | (1 + 3 + 6 + 9 + 6) × 28 = 700 |

在同一台十二卡 Linux 服务器的 ItemGCN 目录、同一训练环境中连续执行：

```bash
bash start_joint_recdcl.sh
bash start_joint_sgl.sh
bash start_joint_stabcf_ahns.sh
bash start_joint_remaining.sh
```

合计十二个后台调度进程、12,936 次训练，默认不跑 sym。组合数用于安排任务，不代表不同模型或数据集的实际耗时相同。

每个文件也接受三个 GPU 编号，顺序始终是 **Ali、Amazon、Yelp2018**。例如分散到不同三卡服务器时，每台都可使用本地卡号 0、1、2：

```bash
# 三个数据集分别指定卡号；应避免与本机其他分组重复占卡
nohup bash start_joint_stabcf_ahns.sh 0 1 2 &
nohup bash start_joint_sgl.sh 3 4 5 &
nohup bash start_joint_remaining.sh 6 7 8 &

# 只检查三个数据集的任务清单，不创建目录、不启动训练
bash start_joint_recdcl.sh --dry-run
bash start_joint_stabcf_ahns.sh 6 7 8 --dry-run

# 自选输出根目录（内部仍按分组/数据集划分）
bash start_joint_remaining.sh 9 10 11 --output-root experiment_results/joint_run2
```

分组输出目录为：

```text
experiment_results/joint_user_repr_no_sym/
    recdcl/{ali,amazon,yelp2018}/
    sgl/{ali,amazon,yelp2018}/
    stabcf_ahns/{ali,amazon,yelp2018}/
    remaining/{ali,amazon,yelp2018}/
```

每个叶目录分别保存 scheduler.log、summary.csv、manifest.json 和训练日志。比如：

```bash
tail -f experiment_results/joint_user_repr_no_sym/recdcl/ali/scheduler.log
tail -f experiment_results/joint_user_repr_no_sym/stabcf_ahns/amazon/scheduler.log
```

支持 `--seed N`、`--base-config FILE`、`--include-sym`、`--dry-run`、`--list-jobs`（需同时 dry-run）及 `--output-root DIR`。添加 --include-sym 后默认输出根目录改为 experiment_results/joint_user_repr。更换 seed 或基础参数应指定新输出根目录。

四个入口共用 `start_joint_user_representation_group.sh`、`start_joint_user_representation.sh` 和 `run_joint_user_representation.py`，上传时需一起保留。不要同时启动旧的“一个数据集跑全部模型”命令和新的分组命令；它们包含重叠的训练任务。分组入口不会自动迁移旧目录结果。

## 单个数据集或单组模型手动启动

Python 入口新增 `--models`，可指定多个模型共享一个调度进程，与原来的 `--model` 互斥：

```bash
python -u run_joint_user_representation.py --dataset ali --gpu_id 6 --models stabcf ahns --output experiment_results/joint_user_repr_no_sym/stabcf_ahns/ali
```

以下是按数据集运行全部模型的原启动方式，适用于不采用上述四组划分时：

在 Linux 服务器的 ItemGCN 目录，激活统一的 Python/PyTorch 环境后：

```bash
# 先预览，验证参数，不训练也不创建输出文件
python run_joint_user_representation.py --dataset ali --gpu_id 0 --output experiment_results/joint_user_repr_no_sym/ali --dry-run

# 三条命令分别启动后台进程，可以连续执行
bash start_joint_user_representation.sh ali 0
bash start_joint_user_representation.sh amazon 1
bash start_joint_user_representation.sh yelp2018 2
```

每个数据集一个调度进程，一次只在指定 GPU 上训练一个配置；不同数据集的调度进程可并行。这是多进程分工，不是一次训练使用多卡。GPU 编号沿用 main.py 的 CUDA_VISIBLE_DEVICES 设置。

启动脚本默认写入 `experiment_results/joint_user_repr_no_sym/<dataset>`，日志追加到该目录的 `scheduler.log`。启动消息表示进程已提交，运行状态以日志为准。再次执行同一命令可续跑。可以传第三个位置参数更改输出路径，额外参数会转交 Python 入口：

```bash
bash start_joint_user_representation.sh ali 0 experiment_results/joint_ali_seed37 --seed 37
bash start_joint_user_representation.sh ali 0 --dry-run

# 自选解释器；默认使用当前环境的 python
PYTHON=/path/to/env/bin/python bash start_joint_user_representation.sh ali 0

tail -f experiment_results/joint_user_repr_no_sym/ali/scheduler.log
```

也可以直接在终端或 tmux 中以前台方式运行（Windows 可用同样的 Python 命令）：

```bash
python -u run_joint_user_representation.py --dataset ali --gpu_id 0 --output experiment_results/joint_user_repr_no_sym/ali
python -u run_joint_user_representation.py --dataset amazon --gpu_id 1 --output experiment_results/joint_user_repr_no_sym/amazon
python -u run_joint_user_representation.py --dataset yelp2018 --gpu_id 2 --output experiment_results/joint_user_repr_no_sym/yelp2018
```

不要将不同数据集、种子或模型子集写入同一输出目录。已有训练调度器使用 `.running.lock` 阻止同目录并发运行；如果进程被强制终止留下锁，先确认旧调度器及训练子进程都已退出，再处理锁文件，不能在仍有进程运行时删除它。

## 搜索空间

完整 B 空间有 29 种，默认排除 sym，实际运行 28 种：degree 的 a,b 均取 0、0.25、0.5、0.75、1，形成 25 个组合（sum、mean、sqrt、sym 为其中四个命名点）；加上 weighted_mean 的 b=0.25、0.5、0.75、1。weighted_mean 的 b=0 与 mean 相同，不重复。排除 sym 同时排除了 degree(a=0.5,b=0.5) 这个等价点，不会以另一个名称重复运行。

| 模型 | 原参数组合数 | 每个数据集联合配置数 |
|---|---:|---:|
| RNS | 1 | 28 |
| MixGCF | 3 | 84 |
| StabCF | 12 | 336 |
| AHNS | 12 | 336 |
| DirectAU | 6 | 168 |
| GraphAU | 9 | 252 |
| SimGCL | 6 | 168 |
| SGL | 24 | 672 |
| RecDCL | 81 | 2268 |
| 合计 | 154 | 4312 |

默认每数据集 4,312 次，三个数据集共 12,936 次；加 --include-sym 后恢复每数据集 4,466 次、总计 13,398 次，默认每配置一个随机种子 2025。模型参数仍来自基础配置文件；没有扩大 lr、dim、层数等公共参数空间。B 从训练交互构建，学习率、层数、损失、早停及评估逻辑沿用现有训练代码。每次训练均从头开始，不继承上一组权重，不保存模型 checkpoint。

## 显式包含 sym 与旧版本续跑

默认启动输出改为 `experiment_results/joint_user_repr_no_sym/<dataset>`，与先前 29 种版本隔离。添加 `--include-sym` 时，后台启动脚本默认输出恢复为 `experiment_results/joint_user_repr/<dataset>`。如果旧版已启动，不能直接把其冻结清单改成 28 种；可保持旧目录使用 `--include-sym` 续跑，或在新目录运行 28 种。

```bash
bash start_joint_user_representation.sh ali 0 --include-sym
```

默认排除 sym 是基于用户已有第一轮结果的明确选择，不代表脚本自动检查或导入了旧结果。更换 seed、数据或基础网格后，如需完整比较，应补跑相应 sym，或使用 --include-sym。

## 预览及调试

```bash
# 展示完整任务 ID 列表
python run_joint_user_representation.py --dataset ali --gpu_id 0 --output experiment_results/joint_user_repr_no_sym/ali --dry-run --list-jobs

# 单独运行一个模型，便于测试；使用独立输出目录
python run_joint_user_representation.py --dataset ali --gpu_id 0 --model mixgcf --output experiment_results/joint_mixgcf_ali

# 指定其他基础网格；相对路径按当前工作目录解析
python run_joint_user_representation.py --dataset ali --gpu_id 0 --base-config experiments.json --output experiment_results/joint_ali_custom
```

## 日志、冻结配置与续跑

每个输出目录包含：

```text
joint_user_representation_config.json  # 展开的联合参数清单
manifest.json                         # 完整默认参数、seed、训练代码摘要
summary.csv                           # 累计任务状态和指标
item_only/<model>/<dataset>/<base_recipe>__B_<construction>/
    status.json
    attempt_001/
        command.json
        config.json
        environment.json
        train.log
        metrics.csv
        result.json
        status.json
```

成功任务自动跳过；失败或中断的任务在下一次启动时新增 attempt_002 等目录重试，保留先前日志。某个任务失败时会继续其他任务，调度进程最终返回非零退出码。更改参数清单、seed 或训练代码需要新输出目录；允许续跑时更换 GPU。训练代码摘要由现有 run_experiments.py 管理。

`summary.csv` 复用配对调度器格式，因此包含 Original 的 pending 占位行；本轮只统计 `phase=item_only`。默认 4,312 条真实任务，不是 8,624 次训练。

本脚本不跨目录自动导入旧实验结果；两批历史日志的 Python/PyTorch/CUDA 版本不同。后续续跑也应使用相同环境，现有 runner 会校验参数、seed 和训练代码，但不会强制校验环境版本或数据文件内容。

训练内部的 epoch 仍按现有验证协议选择。脚本负责完整训练和记录所有候选，不在搜索期间用某个指标筛掉配置，不自动将某一项 test 指标最大值标记为最终模型。
