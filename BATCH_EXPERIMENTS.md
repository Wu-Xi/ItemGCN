# 两轮批量实验

在服务器的 ItemGCN 目录执行。无需安装调度框架，也不依赖相邻的 MixGCF、QRec 等目录。

## 多 GPU：按数据集和形态分成六组（推荐）

默认 save=false，不保存模型参数，保留每次训练的完整日志、指标及最终结果。每组 154 次训练，总计 924 次。六组可同时启动，不必等待 original 完成。

在 ItemGCN 目录执行，下面假设节点有六张可用 GPU；自行修改 gpu_id。六条命令使用同一份配置、同一 seed、同一版本代码，并且必须使用不同 output，避免锁和汇总表冲突。开始后不要修改配置或训练代码。

```bash
mkdir -p scheduler_logs
nohup python -u run_experiments.py --dataset amazon --phase original --gpu_id 0 --output experiment_results/grid02/amazon_original > scheduler_logs/grid02_amazon_original.log 2>&1 &
nohup python -u run_experiments.py --dataset amazon --phase item_only --gpu_id 1 --output experiment_results/grid02/amazon_item_only > scheduler_logs/grid02_amazon_item_only.log 2>&1 &
nohup python -u run_experiments.py --dataset ali --phase original --gpu_id 2 --output experiment_results/grid02/ali_original > scheduler_logs/grid02_ali_original.log 2>&1 &
nohup python -u run_experiments.py --dataset ali --phase item_only --gpu_id 3 --output experiment_results/grid02/ali_item_only > scheduler_logs/grid02_ali_item_only.log 2>&1 &
nohup python -u run_experiments.py --dataset yelp2018 --phase original --gpu_id 4 --output experiment_results/grid02/yelp2018_original > scheduler_logs/grid02_yelp2018_original.log 2>&1 &
nohup python -u run_experiments.py --dataset yelp2018 --phase item_only --gpu_id 5 --output experiment_results/grid02/yelp2018_item_only > scheduler_logs/grid02_yelp2018_item_only.log 2>&1 &
```

如需指定 seed，在六条命令中都加相同的 --seed 整数。可以加 --dry-run 预览。每个进程内部顺序训练，只有进程间并行；不使用多卡分布式训练。少于六张 GPU 时可分批启动，共用 GPU 会竞争显存和算力；六个进程也会分别占用主机内存。

每组目录独立保存 manifest.json、summary.csv 和原有的 phase/model/dataset/组合/attempt_001/train.log。summary.csv 中未运行的另一形态显示 pending，这是该组未负责的任务。train.log 包含 main.py 最后的输出，失败堆栈也保留。成功任务跳过，失败重跑创建新 attempt，不覆盖已有训练日志。再次后台启动时可用 >> 追加调度日志。

带 --dataset 时允许 item_only 独立生成冻结清单，不要求存在 original 结果；各组从同一配置展开相同参数网格。不要使用旧批次结果目录。下面保留不传 --dataset 时的先后两轮模式。

## 启动（先后两轮模式）

先预览任务，不训练、不创建结果目录：

```bash
python run_experiments.py --phase original --output experiment_results/batch01 --dry-run
```

第一轮遍历原始形态：

```bash
python run_experiments.py --phase original --output experiment_results/batch01 --gpu_id 0
```

第一轮全部成功、查看完基准后，手动启动第二轮：

```bash
python run_experiments.py --phase item_only --output experiment_results/batch01 --gpu_id 0
```

不传 --dataset 的先后模式中，两个命令必须指向同一个 `--output`。第二轮复用第一轮**全部参数组合**，不是只取最佳配置。不会自动启动第二轮。

无需设置 seed 列表。需要手动指定时，在第一轮命令加 `--seed 你的整数`；不传则沿用 2025。第二轮省略该参数会继承第一轮。要改 seed，另选结果目录，避免与旧实验混合。

后台运行示例：

```bash
mkdir -p scheduler_logs
nohup python -u run_experiments.py --phase original --output experiment_results/batch01 --gpu_id 0 > scheduler_logs/batch01_original.log 2>&1 &
```

第二轮同理，改成 `--phase item_only` 和另一份调度日志名。每个实验自己的详细日志在结果目录下；调度日志只报告启动、跳过、成功、失败。

## 配置与数量

编辑 `experiments.json`。默认三个数据集、9 个基线，每个数据集 154 组配置：**每轮 462 次，两轮 924 次训练**。不遍历 seed。以下范围均为本次实验选定的搜索网格，不宣称为官方最优配置。

统一设置：dim=64、context_hops=3、batch_size=2048、lr=0.001，所有数据集和两种形态一致。GraphAU 保留原始图对齐机制，graphau_layers=4 包含 0～3 跳对齐项，评分使用传播前表示。SimGCL 保留不聚合初始层的定义。

| 模型 | 搜索网格 | 每个数据集每种形态次数 |
| --- | --- | --- |
| RNS | n_negs=1 | 1 |
| MixGCF | n_negs=[64,32,16] | 3 |
| StabCF | n_negs=[64,32,16] × window_length=[5,10] × alpha=[20.0,21.0] | 12 |
| AHNS | alpha=[0.1,0.5,1] × beta=[0.1,0.4] × n_negs=[16,32]；p=-2、simi=ip | 12 |
| DirectAU | gamma=[0.2,0.5,1,2,5,10] | 6 |
| GraphAU | gamma=[0.2,0.4,1.7] × decaying_base=[0.5,1,1.5] | 9 |
| SimGCL | cl_rate=[0.5,0.2,2] × eps=[0.1,0.5]；ssl_temp=0.2 | 6 |
| SGL | aug_type=[0,1,2] × drop_rate=[0.1,0.4] × cl_rate=[0.1,0.5] × ssl_temp=[0.2,0.5] | 24 |
| RecDCL | bt_coeff=[0.01,0.05,0.1] × poly_coeff=[0.2,1,2] × momentum=[0.1,0.3,0.5] × mom_coeff=[1,5,10] | 81 |

RNS/MixGCF/StabCF/AHNS/SimGCL/SGL：显式 l2=0.001，Adam weight_decay=0。DirectAU/GraphAU/RecDCL：l2=0（当前损失不使用），Adam weight_decay=0.001。RecDCL 的优化器正则作用于全部可训练参数，包括 projector、predictor 和 BatchNorm 参数；历史缓冲区不参与。相同系数不代表不同正则实现强度相同。GraphAU 的 decaying_base 只控制层对齐权重，与参数正则不同。

RecDCL 固定 a=1、degree=4、polyc=1e-7、all_bt_coeff=1。StabCF 的历史数量不包含随后追加的当前正样本。AHNS 保留原公式，分母 s_pos+alpha 接近零时可能产生数值问题；本次不改采样算法。

配置优先级：common → shared → grid 展开后的单组参数。grid 中每个列表做完整笛卡尔积，自动生成包含参数值的唯一目录名（小数点用 p 表示）。也保留原来的 configs=[{id, params}] 格式；同一模型不能同时使用 grid 和 configs。数据集和组合名只能使用字母、数字、下划线、连字符。

自定义配置：

```bash
python run_experiments.py --phase original --config my_experiments.json --output experiment_results/batch02
```

相对 `data_path` 按 ItemGCN 目录解析，当前 loader 要求以 `/` 结尾。第二轮不用再次指定配置文件，它直接读取第一轮保存的清单。

## 两轮之间固定什么

第一轮启动时生成 `manifest.json`，保存所有生效参数（包括 parser 默认值）、唯一 seed、来源说明及训练代码摘要。第二轮直接使用这份清单，只把 gnn 切换为 x 前缀，输出位置随之改变；可重新指定 GPU。

X 形态的 B 固定使用配置中的 `b_mode`，默认 sym。XSGL 不对 B 做增强。XGraphAU 不分配 user embedding table，以 U=BQ 替代用户表，保持 GraphAU 的层加权对齐与均匀性损失；评估用 U 和 Q 点积。

IGCN 不在配对清单中：它采用物品图传播，不能当成 XLightGCN。当前 `xsimgcl` 名字表示本项目“去用户表的 SimGCL”，不是另一篇名为 XSimGCL 的论文模型。

为避免把不同实验混在一起：

- 第一轮续跑时若配置变化，会要求使用新目录。
- 第二轮忽略外部配置文件的变化，固定使用 manifest。
- 训练代码摘要变化时拒绝在同一批次续跑。请在正式跑第一轮前完成手动修复；两轮之间也不要改模型。
- 第一轮有失败或未完成任务时，不启动第二轮；修好运行环境或数据后先续跑 original。

## 日志、失败与恢复

```text
experiment_results/batch01/
  manifest.json
  summary.csv
  original/graphau/ali/gamma-0p4_decaying_base-1p0/
    status.json
    attempt_001/
      status.json
      command.json
      config.json
      environment.json
      train.log
      metrics.csv
      result.json
  item_only/graphau/ali/gamma-0p4_decaying_base-1p0/
    ...
```

`summary.csv` 汇总两轮所有任务，包含全部生效超参数（param_ 前缀）、状态、重试目录、train_log 相对路径、最佳轮次、对应验证/测试指标、耗时。每次任务结束后原子更新汇总表。每个 attempt 的 status.json 保留本次执行状态；父目录的 status.json 指向最新一次执行。`metrics.csv` 记录每次评估的完整精度指标。没有独立验证集时沿用当前 main 的测试指标选最佳轮次，结果的 `selection_split` 会明确写为 test。训练协议未擅自更改。

一个任务失败后继续后面的任务，调度程序最后返回非零退出码。再次执行原命令时，成功任务跳过，失败/中断任务从头训练并写入 `attempt_002` 等新目录，旧日志保留。**这是按实验恢复，不是从中断 epoch 恢复优化器状态。**

正常 Ctrl+C 会停止当前子进程并记录 interrupted；强制杀进程或机器断电可能留下 `.running.lock`，确认旧调度器和训练子进程均已停止后，手动删除该锁再续跑。同一结果目录只允许一个调度器。

## main.py 的改动

继续使用现有 `--gnn`，没有要求重写旧命令。新增 `--seed`、`--run_dir`、`--weight_decay`。weight_decay 是 Adam 权重衰减，与模型损失中的 l2 不同；默认 0 保持旧行为。只有传入 run_dir 才生成结构化实验记录。`if epoch % 1 == 0` 保留。

## 验证

```bash
python -m unittest discover -s tests
```

测试包括 XGraphAU 用户表移除与梯度、各模型训练/评估/保存、两轮参数相等、失败后继续、失败重试保留旧日志、成功跳过、第二轮读取冻结配置、不同 seed 拒绝混用。验证环境为 CPU 小数据，尚未运行服务器完整实验。


## CF 采样基线入口

```bash
python main.py --gnn rns
python main.py --gnn mixgcf --n_negs 32
python main.py --gnn stabcf --n_negs 32 --alpha 20 --window_length 5
python main.py --gnn ahns --n_negs 32 --alpha 1 --beta 0.1 --p -2
```

第二种形态分别使用 xrns、xmixgcf、xstabcf、xahns。保留旧写法 `--gnn lightgcn --ns mixgcf` / `--gnn xlightgcn --ns stabcf`。parser 将前三种名称映射为 LightGCN/XLightGCN 和相应 ns；目录保留采样基线名称，config.json 记录实际 gnn 与 ns。AHNS 保持独立类。

RNS 只采 1 个负物品；其他采样基线采 n_negs 个候选，全部排除当前用户的训练正样本。只有 StabCF 构造 observed_pos_items，沿用历史数量足够时无放回、否则有放回的抽样规则，包含当前正物品的可能性与已有实现一致。StabCF 用混合后的正、负表示计算 BPR；MixGCF 用原正表示和混合负表示。IGCN 当前仍固定 RNS，不属于这组二部图 backbone 的采样对比。

本次配置与旧批次不同，请使用新的输出目录启动。普通 main.py 的默认值保持不变；批量启动器通过命令行显式传入本次全部生效设置。
