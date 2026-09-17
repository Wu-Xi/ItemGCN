import os
import re
import matplotlib.pyplot as plt
import pdb
import pandas as pd
# 设置日志文件目录
log_dir = "/data/yuanzi/codes/CIKM2025/NS4Rec_Framework/logs/WWW2026/0915"
# 0825_2110_LightGCN-ali-bs2048-poolmean-negs32-wind10-alpha10.0.log
# 0907_2145_LightGCN-ali-bs2048-poolmean-negs32-wind5-alpha21.0-beta0.0
# 0908_1928_LightGCN-amazon-bs2048-poolmean-negs64-wind10-alpha1.0-beta0.75
log_string = "_LightGCN-yelp2018-bs2048-poolmean-negs64-wind5-alpha"

# apegnn-ali-lr_1e3-l2_1e3-batch_size2048-window_length7-negs16-alpha1.0-tau0.1-poolsum.log


alpha_pattern = re.compile(r'alpha([0-9.]+)')

# 匹配 recall/ndcg/precision/hit_ratio 的最后一行结果
metric_pattern = re.compile(
    r"recall\s+\|\s+\[([0-9.,\s]+)\].*?"
    r"ndcg\s+\|\s+\[([0-9.,\s]+)\].*"
    r"precision\s+\|\s+\[([0-9.,\s]+)\].*?"
    r"hit_ratio\s+\|\s+\[([0-9.,\s]+)\]",
    re.DOTALL
)

# 匹配所有 metric 行（用于提取最后一次）
row_pattern = re.compile(
    r"recall\s+\|\s+\[([0-9.,\s]+)\]\s*\|"
    r"\s*\[([0-9.,\s]+)\]\s*\|\s*\[([0-9.,\s]+)\]\s*\|\s*\[([0-9.,\s]+)\]"
)
# 储存每个文件的 recall@20 数值列表
recall_data = {}
# pdb.set_trace()
log_list = []
#######################################################
# 方式一：遍历目录下所有 .log 文件
for filename in os.listdir(log_dir):
    if log_string not in filename:
        continue
    
    if filename.endswith(".log"):
        filepath = os.path.join(log_dir, filename)
        log_list.append(filepath)
#######################################################
#######################################################
# # 方式二：正则化遍历目录下所有 .log 文件
# apegnn-yelp2018-lr_1e4-l2_1e4-batch_size2048-window_length4-negs16-alpha11.0-tau0.1-poolsum.log
# file_pattern = re.compile(r'apegnn-yelp2018-lr_1e4-l2_1e4-batch_size2048-window_length[0-9]+(?:\.[0-9]+)?-negs16-alpha11\.0-tau0\.1-poolsum.*\.log$')
# for filename in os.listdir(log_dir):
#     if not file_pattern.search(filename):
#         continue

#     filepath = os.path.join(log_dir, filename)
#     log_list.append(filepath)
#######################################################
# 提取 alpha 值的函数
def extract_alpha(file_path):
    # 假设文件名包含 alpha 后面接数字，如 alpha0.01
    # match = re.search(r'window_length([0-9]+(?:\.[0-9]+)?)-negs32-alpha21\.0-tau0\.1',file_path)
    match = re.search(r'alpha([0-9]+(?:\.[0-9]+)?)', file_path)
    if match:
        return float(match.group(1))
    return float('inf')  # 如果提取不到，排到最后
# ...existing code...

def extract_tau(file_path):
    match = re.search(r'tau([0-9]+(?:\.[0-9]+)?)', file_path)
    if match:
        return float(match.group(1))
    return float('inf')  # 如果提取不到，排到最后

# 按 tau 从小到大排序
# log_list.sort(key=extract_tau)

# ...后续代码不变...
# 按 tau 从小到大排序
log_list.sort(key=extract_alpha)

# 输出检查排序结果（可选）
# for log_file in log_list:
#     print(log_file)

ndcg_list = []  # 新增 NDCG@20 列

label_list, recall_list, ndcg_list = [], [], []
for filepath in log_list:
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.readlines()
    label_list.append(extract_tau(filepath))
    for i in range(1, 5, 1):
        try:
            matches = re.findall(r"\[([^\]]+)\]", content[-2])
        except Exception as e:
            print(e)
        if matches and len(matches) >= 2:
            recall_values = [float(x.strip()) for x in matches[0].split(",")]
            ndcg_values = [float(x.strip()) for x in matches[1].split(",")]
            if len(recall_values) >= 2 and len(ndcg_values) >= 2:
                recall_at_20 = recall_values[1]
                ndcg_at_20 = ndcg_values[1]
                recall_list.append(recall_at_20)
                ndcg_list.append(ndcg_at_20)
                file_name = filepath.split("/")[-1]
                print(f"tau={extract_tau(filepath):.2f} RECALL@20: {recall_at_20}, NDCG@20: {ndcg_at_20}, {file_name}")
                break
            else:
                print("RECALL/NDCG 列中没有足够的值。")
        else:
            print("没有找到指标数据。")

# # 保存到 Excel
# df = pd.DataFrame({
#     'tau': label_list,
#     'RECALL@20': recall_list,
#     'NDCG@20': ndcg_list
# })
# df.to_excel('tau_metrics.xlsx', index=False)
