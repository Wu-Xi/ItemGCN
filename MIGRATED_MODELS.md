> 采样接口已更新：LightGCN/XLightGCN 的 ns 现在实际支持 rns、mixgcf、stabcf；仅 StabCF 采历史正样本。详见 BATCH_EXPERIMENTS.md，以下固定 RNS 的描述为历史记录。

> 当前统一实验设置以 BATCH_EXPERIMENTS.md 和 experiments.json 为准：3 层、batch 2048、lr/l2=0.001、Adam weight_decay=0。下文原迁移默认值为历史记录，已被统一配置覆盖。

# SimGCL、SGL、RecDCL 的两种形态

这三个模型从本地 MixGCF 迁入，ItemGCN 可以独立运行，不需要把 MixGCF 上传到服务器。

| 原始形态 `--gnn` | 无用户嵌入表形态 `--gnn` | 负采样 |
| --- | --- | --- |
| simgcl | xsimgcl | 每条交互一个负物品 |
| sgl | xsgl | 每条交互一个负物品 |
| recdcl | xrecdcl | 不需要 |

## 使用

在 ItemGCN 目录运行，以下六条分别对应六种形态：

```bash
python main.py --gnn simgcl --dataset yelp2018 --gpu_id 0
python main.py --gnn xsimgcl --dataset yelp2018 --gpu_id 0
python main.py --gnn sgl --dataset yelp2018 --gpu_id 0
python main.py --gnn xsgl --dataset yelp2018 --gpu_id 0
python main.py --gnn recdcl --dataset yelp2018 --gpu_id 0
python main.py --gnn xrecdcl --dataset yelp2018 --gpu_id 0
```

默认学习率 0.001、batch_size 2048。模型相关默认值如下，均可显式覆盖：

| 参数 | SimGCL / XSimGCL | SGL / XSGL | RecDCL / XRecDCL |
| --- | --- | --- | --- |
| dim | 64 | 64 | 2048 |
| context_hops | 2 | 3 | 2 |
| l2 | 0.0001 | 0.001 | 不使用 |
| cl_rate | 0.5 | 0.1 | 不使用 |
| ssl_temp | 0.2 | 0.2 | 不使用 |
| eps | 0.1 | 不使用 | 不使用 |
| drop_rate | 不使用 | 0.1 | 不使用 |
| aug_type | 不使用 | 1 | 不使用 |

SGL：`--aug_type 0` 为节点丢弃，`1` 为边丢弃，`2` 为每层独立丢边。两张增强图每轮重新生成，当轮复用。

RecDCL：默认 `--encoder lightgcn`，也支持 `--encoder mf`。损失参数沿用本地实现：`a=1`、`polyc=1e-7`、`degree=4`、`poly_coeff=0.2`、`bt_coeff=0.01`、`all_bt_coeff=1`、`mom_coeff=10`、`momentum=0.3`。2048 维会比 64 维消耗更多显存；可通过 `--dim` 调整，但这会改变实验设定。

保存时仍沿用当前入口的 `model_.ckpt`，不同实验应指定不同的 `--out_dir`，例如 `--save true --out_dir weights/xsgl/`。

## X 形态的定义

沿用当前 XLightGCN / XDirectAU 的做法：初始物品嵌入为可训练表 Q，初始用户嵌入为 U = BQ，不分配可训练的 user embedding table。后续传播、损失和评分保持对应模型的定义。用户侧损失可以通过 BQ 对 Q 求导。

B 复用现有 `build_B`，默认 `--b_mode sym`，即 `D_U^(-1/2) R D_I^(-1/2)`。也可使用现有 `mean`、`sum`、`sqrt`、`degree`、`weighted_mean` 模式，以及 `--b_a`、`--b_b`。仅使用训练交互构图。

XSGL 的 B 固定从完整训练交互构造，两个视图共享 U = BQ，图增强只作用于后续传播邻接矩阵。它不是“每个视图使用丢弃后的交互重新构造 B”的变体。

XRecDCL 保留用户历史目标 `u_target_his`，这是无梯度的 buffer，不是可训练用户表，但仍占用用户数 × 维度的内存。投影器和预测器也继续训练；“无用户嵌入表”不等于只有物品表一个可训练参数。

## 与现有参数的关系

- 六个新增模型使用用户—物品二部图。`igcn` 的物品图 `s_mode/s_norm/s_top_k/...` 不作用于它们。
- SimGCL 的层平均排除初始层；SGL 和 RecDCL 的 LightGCN 编码器包含初始层。为保持模型定义，新增模型不使用现有 `embedding0` 和 `pool` 开关。
- SimGCL/SGL 保留本地 QRec 迁移版的求和 BPR、传播后嵌入 L2，以及对比损失。SGL 使用用户和物品拼接后的联合对比损失。它们不使用 `ns`、`n_negs`、StabCF 历史采样或混合负采样，每次只采一个排除训练正样本的负物品。
- RecDCL 不生成负样本，也不为负采样检查“用户是否交互了全部物品”。
- 新模型不使用现有 edge/mess dropout 参数；SimGCL/SGL 分别通过自身的 eps/drop_rate 控制增强。
- 最后不足 batch_size 的 batch 保留。RecDCL 单样本尾批的 BatchNorm 使用已有运行统计。

## 修改与验证

新增 `modules/SimGCL.py`、`modules/SGL.py`、`modules/RecDCL.py`、`modules/graph_contrastive.py`。`modules/LightGCN.py` 仅追加六个类的导入，仍可以从该文件导入它们。

`main.py` 注册六个模型，按模型决定负采样数量，调用 SGL 每轮更新钩子，并修复 `get_feed_dictv4` 在定义 `entity_pairs` 前使用它的问题。`utils/data_loader.py` 增加二部图/B 构图分支；`utils/parser.py` 增加参数及默认值。原有 lightgcn/igcn 的参数默认值保持不变，`if epoch % 1 == 0` 保留。

```bash
python tests/test_migrated_models.py
```

CPU 测试覆盖六种形态、SGL/XSGL 各三种增强，共十组实际入口的两轮训练、评估与保存；另检查三个 X 模型不存在用户表参数、BQ 的值与梯度、单样本尾批、检查点恢复。尚未执行服务器 GPU 或完整数据集实验，不能据此判断推荐指标优劣。

## 已有模型接口修复

入口还支持 `xlightgcn`、`ahns`、`xahns`、`directau`、`xdirectau`、`graphau`。LightGCN/XLightGCN/IGCN 固定使用一个随机负样本，不根据 `ns` 切换算法，也不生成历史正样本。AHNS/XAHNS 使用 `n_negs` 个候选自行筛选。

AHNS 新增命令行参数 `simi=ip`、`p=-0.5`、`beta=0.5` 作为可运行的默认配置，正式实验请显式指定你需要的参数，这些默认值不代表已经调优。DirectAU/XDirectAU 的 gamma 缺省为 1，GraphAU 缺省为 0.4；GraphAU 默认 graphau_layers=4、decaying_base=1.4。

评估返回完整精度，表格显示五位小数。最佳模型只在指标严格提升时更新；持平也累计早停次数。仍按原入口的第二个 K 选模型。

回归测试已扩展到上述已有模型，共十八组两轮入口训练，并检查持平早停、微小提升和首轮零分。
