# 用户活跃度四组收益分析

目的：检验 Item-only 相比同 backbone 的 Original，在哪类用户上获益最多。
使用 `best54_configs.json` 冻结的 3 数据集 × 9 backbone × 2 版本，以及各任务
`status.json` 指向的成功 checkpoint。每个 checkpoint 仅推理；四组共享同一模型。

## 分组和收益口径

默认 `equal_freq` 模式：每个数据集仅纳入有测试正例的用户，按 `(训练集不同交互物品数, 数值用户 ID)`
升序排列，分成四个近似等人数的组。G1 最不活跃，G4 最活跃。相同度数可能跨组；
组间人数最多相差 1。所有模型使用同一份数据和分组，报告每组人数及度数范围。
分组不使用验证集或测试集交互次数。

保持原有全物品排名与训练交互屏蔽规则，计算每位用户的 Recall、NDCG、Hit@10/20/50，
再在组内做用户平均。为了对齐已有整体评估，复用其分数置换和 `torch.topk` 口径。
该置换后的索引不能当作真实 item ID，因此本入口专门做用户分组。

对每个 dataset/backbone/metric/K/group：

- `absolute_gain = Item-only - Original`：主要用于判断哪组获益更多；负值为退步。
- `relative_gain_pct = 100 × absolute_gain / Original`：相对提升；基线为 0 时留空。
- `overall_gain_contribution = 该组人数 / 总测试人数 × absolute_gain`：对总体宏平均提升的贡献。
  四组贡献之和等于总体指标之差，负贡献照实保留。
- `g1_minus_g4_absolute_gain = G1绝对提升 - G4绝对提升`：正值说明低活跃组的绝对提升更大。

主图显示 Recall@20、NDCG@20 的绝对提升（乘 100 后为百分点）。不要把 Item-only 本身
分数最高的组当成收益最大的组，也不要仅凭相对百分比判断。报告保留并列和全组无正收益的情形。
这些是固定 checkpoint 的描述性比较；不代表多 seed 显著性检验，也不能单独证明因果机制。

## 服务器准备

在服务器 ItemGCN 仓库中同步以下代码：

- `evaluate_user_activity.py`
- `utils/user_activity.py`
- 修改后的 `utils/checkpoint.py`
- `tests/test_user_activity.py`（验证用）

保留现有模型代码、`best54_configs.json`、原始 `data/` 和完整 checkpoint 目录。
使用原训练环境（记录为 Python 3.12 / PyTorch 2.9），含 NumPy、SciPy；
绘图另需 matplotlib。加载采用 `torch.load(..., weights_only=True)`。
本地旧版 PyTorch 不用于实际加载这批 checkpoint。

先验证计算逻辑：

```bash
python -m unittest discover -s tests -p test_user_activity.py -v
```

## 优先评估服务器原始 FP32 checkpoint

以下路径对应训练记录中的目录；如实际存放位置不同，替换 `--checkpoint-root`。

```bash
python evaluate_user_activity.py \
  --checkpoint-root checkpoints/best54_20261016_24way \
  --output-root experiment_results/user_activity_fp32 \
  --dry-run

python -u evaluate_user_activity.py \
  --checkpoint-root checkpoints/best54_20261016_24way \
  --output-root experiment_results/user_activity_fp32 \
  --gpu-id 0 --batch-size 256 --plot
```

一张 GPU 顺序评测全部 54 个模型；每个模型独立子进程，完成后释放其资源。
`--dry-run` 检查配置、数据指纹和全部权重指纹，不创建评测输出或执行模型。
真实权重的结构与加载仍由正式运行检验。

可先加 `--dataset ali --models rns` 跑一对，使用独立输出目录，例如
`--output-root experiment_results/user_activity_smoke`。确认结果后再用上面的全量命令。

## 如只使用 FP16 分享包

```bash
python -u evaluate_user_activity.py \
  --checkpoint-root logs/ItemGCN_best54_checkpoint/checkpoints/best54_20261016_24way_fp16 \
  --fp16-manifest logs/ItemGCN_best54_checkpoint/checkpoints/best54_20261016_24way_fp16/FP16_CONVERSION_MANIFEST.json \
  --output-root experiment_results/user_activity_fp16 \
  --gpu-id 0 --batch-size 256 --plot
```

可以先在同一命令末尾追加 `--dry-run`。转换包中的 `checkpoint.json` 仍保存源 FP32
哈希，因此必须显式提供 `--fp16-manifest`。加载器核对转换清单中的路径/文件大小、
FP16 张量类型和有限性，以及模型 state_dict 的键和形状；数据指纹仍严格校验。
清单没有 FP16 哈希，文件大小不构成内容真实性验证；实际 SHA256 会写入评测记录以便追溯。
原有无 `--fp16-manifest` 的加载入口保持严格权重哈希校验。

FP16 存储的数值载入 FP32 模型做推理，转回 FP32 不会恢复已丢失精度。报告记录存储格式和
推理类型，并比较重新计算的总体指标与该 checkpoint 自身的源 FP32 指标。
默认任一可对齐指标绝对差超过 `1e-4` 就标记 `review_required`；这是人工复核阈值，
不是精度等价保证。可用 `--drift-tolerance` 在运行前明确修改。不要拿历史搜索表替代本次结果。

## 输出与恢复

- `group_gains.csv`：每组 Original、Item-only、绝对/相对提升、总体贡献及人数/度数范围。
- `summary.json`：完整结果和每对模型获益最多的组、G1−G4 提升差。
- `REPORT.md`：Recall@20/NDCG@20 的摘要，保留负收益、并列及待复核标记。
- `*_user_gain.png/svg`：使用 `--plot` 生成的按 backbone 分面的收益柱状图。
- `<dataset>/<model>/<phase>/evaluation.json`：配置来源、epoch、总体/组内指标、精度差异和环境。
- 同目录 `users.npz`：按用户 ID 排序的每用户指标、用户组、训练度数；可用于后续配对统计。
- 同目录 `evaluation.log`：该模型加载和推理日志。
- `evaluation_manifest.json`：本次代码、输入权重和运行设置的指纹。

相同命令重跑会校验并跳过已完成结果，接着完成剩余任务。改变配置、checkpoint、代码、
batch size、CPU/GPU 模式或筛选范围时使用新输出目录。正常退出会释放 `.evaluation.lock`；
若被强制杀死，先确认旧进程已结束，再手动处理该锁。

已评测完后，可在原命令加 `--summarize-only --plot` 重新生成汇总/图（保持其余设置不变）。
如缺少 matplotlib，先去掉 `--plot` 完成指标计算，之后在同环境补上绘图库再汇总。
使用多个 GPU 时，可对三个数据集分别指定不同 `--dataset`、`--gpu-id` 和独立输出目录。


## 三种分组策略

- `--group-method equal_freq`：默认等人数分组，按训练度数和数值用户 ID 排序后四等分。
- `--group-method equal_width`：将包含两端的整数度数范围 `[最小度数, 最大度数]`
  划为四段，各段包含的整数个数最多相差 1；相同度数用户留在同一组。
- `--group-method fixed --degree-thresholds 5 20 50`：G1 为 ≤5，G2 为 6–20，
  G3 为 21–50，G4 为 ≥51。三个阈值必须为非负、严格递增的整数。

三种方式都只统计测试用户的不同训练交互物品数。等宽不是等人数，容易受最大度数的影响。
例如最小度数为 2、最大度数为 11，等宽范围为 2–4、5–6、7–9、10–11。
范围只有两个整数 0、1 时，G1 分到度数 0，G3 分到度数 1，G2/G4 的整数区间为空；
所有度数相同时，全部用户进入 G1，其余组为空。这是整数范围划分的明确约定。

等宽和固定阈值方式可能出现空组。空组的观测度数统计、指标、绝对/相对收益均为 JSON
`null`，CSV 对应字段留空，对总体收益的贡献为 0，图中标注 `n=0`。
若 G1 或 G4 为空，G1−G4 收益差记为 `N/A`。获益最大组的比较会排除空组。
`REPORT.md` 同时展示分配区间、实际观测度数范围和人数。

本地原始数据上核对的四组测试用户数如下（2026-10-18；不是模型性能结果）：

| 数据集 | 策略 | G1 | G2 | G3 | G4 |
|---|---|---:|---:|---:|---:|
| Ali | equal_freq | 18131 | 18131 | 18131 | 18131 |
| Ali | equal_width | 72327 | 185 | 10 | 2 |
| Ali | fixed 5/20/50 | 30222 | 39342 | 2811 | 149 |
| Amazon | equal_freq | 34237 | 34236 | 34236 | 34236 |
| Amazon | equal_width | 136789 | 134 | 16 | 6 |
| Amazon | fixed 5/20/50 | 61355 | 70424 | 4519 | 647 |
| Yelp2018 | equal_freq | 7917 | 7917 | 7917 | 7917 |
| Yelp2018 | equal_width | 31634 | 30 | 2 | 2 |
| Yelp2018 | fixed 5/20/50 | 0 | 10842 | 14984 | 5842 |

等宽组的人数非常不均衡；极少数用户组的收益不能当作稳定的群体结论。
Yelp2018 固定阈值的 G1 为空，所以该方案无法给出 G1−G4 比较。
可将等人数方案用于主分析，另两种作为补充，并完整报告人数；不要根据模型收益重新选阈值。

三种方式分别使用独立输出目录；代码更新后也使用新的输出目录。旧版本结果不能直接用
新版本 `--summarize-only` 读取。`summary.json` 和运行清单均记录分组方式与固定阈值。

### 服务器运行三种方式

先在服务器仓库中 `git pull --ff-only origin main`，激活原评测环境并运行上文单元测试。
以下变量使用之前服务器报错中确认的 FP16 目录；如果已移动模型，修改这一行。
保留整个转换包及其中的 `FP16_CONVERSION_MANIFEST.json`。

```bash
ACTIVITY_CKPT_ROOT="data/checkpoints/best54_20261016_24way_fp16"

# 三种方式各对 54 个 checkpoint 做预检；遇错退出此循环。
for method in equal_freq equal_width fixed; do
  python evaluate_user_activity.py \
    --checkpoint-root "$ACTIVITY_CKPT_ROOT" \
    --fp16-manifest "$ACTIVITY_CKPT_ROOT/FP16_CONVERSION_MANIFEST.json" \
    --output-root "experiment_results/activity_v2_${method}" \
    --group-method "$method" --degree-thresholds 5 20 50 --dry-run || break
done
```

三种预检全部通过后，先对 Ali/RNS 的两个 checkpoint 分别试跑三种方式：

```bash
for method in equal_freq equal_width fixed; do
  python -u evaluate_user_activity.py \
    --checkpoint-root "$ACTIVITY_CKPT_ROOT" \
    --fp16-manifest "$ACTIVITY_CKPT_ROOT/FP16_CONVERSION_MANIFEST.json" \
    --output-root "experiment_results/activity_v2_smoke_${method}" \
    --group-method "$method" --degree-thresholds 5 20 50 \
    --dataset ali --models rns --gpu-id 0 --batch-size 256 --plot || break
done
```

查看三份 `REPORT.md` 后，在一张 GPU 上顺序启动三种方式的完整评测：

```bash
nohup bash -c '
  set -e
  for method in equal_freq equal_width fixed; do
    python -u evaluate_user_activity.py \
      --checkpoint-root "$1" \
      --fp16-manifest "$1/FP16_CONVERSION_MANIFEST.json" \
      --output-root "experiment_results/activity_v2_${method}" \
      --group-method "$method" --degree-thresholds 5 20 50 \
      --gpu-id 0 --batch-size 256 --plot
  done
' bash "$ACTIVITY_CKPT_ROOT" >> activity_v2_all.log 2>&1 < /dev/null &
echo $!
```

这里每种方式各执行 54 次 checkpoint 推理，共 162 次，按顺序执行。
各方式结果分别位于 `activity_v2_equal_freq`、`activity_v2_equal_width`、`activity_v2_fixed`。
用 `tail -f activity_v2_all.log` 查看总进度，具体推理进度查看模型子目录的 `evaluation.log`。
后台启动不代表评测完成；任一方式失败时循环停止，应先查看日志。
