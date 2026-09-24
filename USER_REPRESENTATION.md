# 用户表征构建实验

运行 run_user_representation.py，默认覆盖配置中的所有九个 X 基线，每个模型固定 experiments.json 中第一组模型超参数（不是自动选择最佳），仅遍历 U=BQ 的 B。启动时打印固定参数，完整配置保存到 user_representation_config.json。若要采用已确定的最佳参数，使用 --base-config 指定每个模型只含一组参数的配置文件。

默认 b_a、b_b 候选为 [0,0.25,0.5,0.75,1]。degree 使用 D_U^(-a) R D_I^(-b)，25 个组合，其中四个用 sum、mean、sqrt、sym 名称标记；加 weighted_mean 的 b=[0.25,0.5,0.75,1] 共 29 组。weighted_mean 的 b=0 与 mean 重复，已去掉。每数据集 9×29=261 次，三个数据集 783 次。

```bash
python run_user_representation.py --dataset amazon --gpu_id 1 --output experiment_results/user_repr_amazon
python run_user_representation.py --dataset ali --gpu_id 1 --output experiment_results/user_repr_ali
python run_user_representation.py --dataset yelp2018 --gpu_id 2 --output experiment_results/user_repr_yelp2018
```

加 --dry-run 预览，不创建目录。加 --exponents 0 0.5 1 缩小指数网格（仍保留四个固定对照）。--model rns 可只跑 XLightGCN-RNS；其他名称同 experiments.json，可用 --trial 精确选择该模型的固定组合 ID。--seed 默认2025，三个进程可手动设为相同值。只运行 X 形态，不需先跑 original。

不保存模型参数。复用现有调度器：每组合独立 train.log、metrics.csv、result.json，根目录 summary.csv 汇总结果；成功跳过、失败创建新 attempt 重试。不同数据集使用不同 output。配置变化必须使用新目录。此前讨论的隐藏命令行参数功能尚未实现，此入口仍复用原命令行传参方式。

沿用基础配置的维度、层数、优化器等公共设置，不修改模型和 B 构造公式，不遍历物品图 s_* 参数。
