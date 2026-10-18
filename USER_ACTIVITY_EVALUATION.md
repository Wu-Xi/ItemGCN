# 用户活跃度四组收益分析

目的：检验 Item-only 相比同 backbone 的 Original，在哪类用户上获益最多。
使用 `best54_configs.json` 冻结的 3 数据集 × 9 backbone × 2 版本，以及各任务
`status.json` 指向的成功 checkpoint。每个 checkpoint 仅推理；四组共享同一模型。

## 分组和收益口径

每个数据集仅纳入有测试正例的用户，按 `(训练集不同交互物品数, 数值用户 ID)`
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
