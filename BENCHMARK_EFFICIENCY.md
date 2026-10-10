# 从头训练补测54组配置的效率

`benchmark_efficiency.py` 独立读取根目录 `best54_configs.json`。不依赖checkpoint、历史日志或analysis目录，不修改正式训练代码，不保存模型权重。每组从seed=2025随机初始化，默认完整训练1轮预热、3轮测量，每组单独子进程，顺序完成后退出并释放显存。

## 服务器运行

在ItemGCN目录激活原训练环境。以下以空闲的GPU 6为例，替换成实际空闲卡号。

```bash
# 检查54组参数和数据指纹，不初始化GPU、不创建输出目录
python benchmark_efficiency.py --gpu-id 6 --dry-run

# 先测试Ali/RNS的一对Original与Item-only：共2个任务，每个4轮
python -u benchmark_efficiency.py --gpu-id 6 --dataset ali --model rns

# 相同卡、环境、配置和输出目录下，继续全部54组，自动跳过上述成功任务
nohup python -u benchmark_efficiency.py --gpu-id 6 > efficiency54.out 2>&1 < /dev/null &

# 查看调度日志
tail -f efficiency54.out
```

默认输出 `experiment_results/efficiency54/`，已被现有gitignore忽略。`summary.csv` 每完成一项便更新，可导入Excel，并填写效率LaTeX表。`tail -f`用Ctrl+C退出查看，不会停止nohup任务。

其他筛选方式：

```bash
# 单个数据集，两版本共18组
python -u benchmark_efficiency.py --gpu-id 6 --dataset amazon

# 单个模型、单个版本
python -u benchmark_efficiency.py --gpu-id 6 --dataset ali --model recdcl --phase item_only

# 修改测量轮数：使用新输出目录，不能混入默认1+3协议的结果
python -u benchmark_efficiency.py --gpu-id 6 --warmup-epochs 2 --measure-epochs 5 \
  --output-root experiment_results/efficiency54_w2_m5
```

支持 `--config` 指定其他选定配置JSON，`--data-path`指定数据根目录。JSON结构沿用best54_configs.json；程序校验数据文件指纹。

若分配多张卡，每个调度器使用不同的输出目录，例如分别运行Ali、Amazon、Yelp2018；一对Original和Item-only保持在同一张卡。避免多个调度器对同一输出目录并发写入。最终合并前核对GPU型号和软件环境。即使GPU同型号，其他任务竞争CPU和内存带宽也会影响速度。

## 测量口径

- **Trainable Params**：实际`requires_grad=True`的参数元素总数；M=个数/1,000,000。包含投影器等额外网络参数，不包含buffer及稀疏图。
- **Peak Memory**：模型和图加载完成后、第一轮训练前重置计数，统计预热及测量期间的`torch.cuda.max_memory_allocated()`；GiB=字节/1024^3。包含仍存活的模型、图、梯度、Adam状态和中间张量；不减去基线显存。它不包含CUDA上下文等所有设备开销，不能等同nvidia-smi的进程显存。额外保留peak reserved用于诊断。
- **Time/Epoch**：预热后的完整epoch耗时均值；标准差采用样本标准差（ddof=1），只有1个测量epoch时记0。默认3个连续epoch不是3个随机种子。
- 计时包含每轮模型hook、SGL图增强、负采样、历史物品采样、batch传输、前向、损失检查、反向、Adam更新、轮末loss读取；两端执行CUDA同步。与main.py的时间边界一致，打乱训练交互在计时区间之外。数据加载、初次构图、模型初始化、评价、保存和日志输出不计入Time/Epoch。
- 预热不计入时间均值，但计入训练峰值显存，以覆盖第一次Adam更新及初始化训练状态的开销。每组模型进程独立，不在batch间清空显存缓存。
- 模型、损失、采样、batch size等沿用各自选中的最佳配置；只改变总训练轮数并关闭评价/保存。若两个版本最优负样本数等不同，时间差也包含这些配置差异，不能全部归因于去掉用户表。
- 每轮执行完整训练集，记录实际batch数和交互条数。测量的是初期短程训练成本；没有最佳epoch、收敛总时间或准确率，不能把4轮当作模型收敛轮数。
- 新建Adam优化器并实际更新参数；不读取、不覆盖任何正在生成的正式checkpoint。

## 输出文件

```text
experiment_results/efficiency54/
  manifest.json                         # 配置摘要、代码摘要、协议、硬件软件环境
  summary.csv                           # 已执行任务总表，失败项保留状态，指标留空
  ali/rns/original/
    manifest.json                       # 选中配置与实际补测设置
    status.json                         # 当前attempt和完成状态
    attempt_001/
      request.json                      # 工作进程完整输入
      command.json
      train.log
      status.json
      result.json                       # 成功时：三个指标、各轮耗时、参数与环境
```

主表填写 `params_m`、`peak_allocated_gib`、`time_epoch_mean_s`（可加`time_epoch_std_s`）。总参数量是`trainable_params`，每轮详情是result.json的`epoch_records`。GPU型号、容量、可用时的UUID、PyTorch/CUDA/cuDNN版本、CPU线程配置均保留。

重复运行只跳过匹配配置的成功结果。失败或结果缺失会新建attempt重新从头补测；旧日志保留。报错后先看对应attempt的train.log。显存不足不会自动降低batch size，以免改变比较口径。代码、数据、GPU、环境或测量协议改变时需换新输出目录。中断后如果留下`.benchmark.lock`，先确认该调度器和子进程均已结束，再删除该锁文件。

`--cpu`仅用于微型数据的功能测试；GPU显存字段为null/CSV空白，不能作为论文GPU效率结果。正式GPU任务如果CUDA不可用会在预检查时停止，不会静默改用CPU。

脚本只新增仓库根目录文件，复用现有main.py中的采样函数、utils中的构图及模型工厂，不改变现有54组正式训练的代码摘要。
