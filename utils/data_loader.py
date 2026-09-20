import numpy as np
import scipy.sparse as sp
import torch
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

n_users = 0
n_items = 0
dataset = ''
train_user_set = defaultdict(list)
test_user_set = defaultdict(list)
valid_user_set = defaultdict(list)


def read_cf_amazon(file_name):
    return np.loadtxt(file_name, dtype=np.int32)  # [u_id, i_id]

def read_cf_yelp2018(file_name):
    inter_mat = list()
    lines = open(file_name, "r").readlines()
    for l in lines:
        tmps = l.strip()
        inters = [int(i) for i in tmps.split(" ")]
        u_id, pos_ids = inters[0], inters[1:]
        pos_ids = list(set(pos_ids))
        for i_id in pos_ids:
            inter_mat.append([u_id, i_id])
    return np.array(inter_mat)

def swap_train_sp_mat(valid_user_set, test_user_set):
    # valid
    valid_uid2swap_idx = np.array([None] * n_users)
    valid_uid2rev_swap_idx = np.array([None] * n_users)
    # valid_pos_len_list = np.zeros(n_users)
    # test_pos_len_list = np.zeros(n_users)
    valid_pos_len_list = []
    for uid in valid_user_set:
        positive_item = valid_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        valid_uid2swap_idx[uid] = swap_idx
        valid_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # valid_pos_len_list[uid] = postive_item_num
        valid_pos_len_list.append(postive_item_num)
    # test
    test_uid2swap_idx = np.array([None] * n_users)
    test_uid2rev_swap_idx = np.array([None] * n_users)
    test_pos_len_list = []
    for uid in test_user_set:
        positive_item = test_user_set[uid]
        postive_item_num = len(positive_item)
        swap_idx = torch.FloatTensor(sorted(set(range(postive_item_num)) ^ set(positive_item)))
        test_uid2swap_idx[uid] = swap_idx
        test_uid2rev_swap_idx[uid] = swap_idx.flip(0)
        # test_pos_len_list[uid] = postive_item_num
        test_pos_len_list.append(postive_item_num)

    return (valid_uid2swap_idx, valid_uid2rev_swap_idx, valid_pos_len_list), \
            (test_uid2swap_idx, test_uid2rev_swap_idx, test_pos_len_list)

def statistics(train_data, valid_data, test_data):
    global n_users, n_items
    n_users = max(max(train_data[:, 0]), max(valid_data[:, 0]), max(test_data[:, 0])) + 1
    n_items = max(max(train_data[:, 1]), max(valid_data[:, 1]), max(test_data[:, 1])) + 1

    if args.dataset not in ['yelp2018','movielens','gowalla']:
        n_items -= n_users
        # remap [n_users, n_users+n_items] to [0, n_items]
        train_data[:, 1] -= n_users
        valid_data[:, 1] -= n_users
        test_data[:, 1] -= n_users

    for u_id, i_id in train_data:
        train_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in test_data:
        test_user_set[int(u_id)].append(int(i_id))
    for u_id, i_id in valid_data:
        valid_user_set[int(u_id)].append(int(i_id))

    train_sp_mat = sp.csr_matrix((np.ones_like(train_data[:, 0]),
                                    (train_data[:, 0], train_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    valid_sp_mat = sp.csr_matrix((np.ones_like(valid_data[:, 0]),
                                    (valid_data[:, 0], valid_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    test_sp_mat =sp.csr_matrix((np.ones_like(test_data[:, 0]), 
                                    (test_data[:, 0], test_data[:, 1])), dtype='float64', shape=(n_users, n_items))
    # prepare for top k accerate
    valid_pre, test_pre = swap_train_sp_mat(valid_user_set, test_user_set)
    return train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre

def build_sparse_graph(data_cf):
    def _bi_norm_lap(adj):
        # D^{-1/2}AD^{-1/2}
        rowsum = np.array(adj.sum(1))

        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

        bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        return bi_lap.tocoo()

    def _si_norm_lap(adj):
        # D^{-1}A
        rowsum = np.array(adj.sum(1))

        d_inv = np.power(rowsum, -1).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat_inv = sp.diags(d_inv)

        norm_adj = d_mat_inv.dot(adj)
        return norm_adj.tocoo()

    cf = data_cf.copy()
    cf[:, 1] = cf[:, 1] + n_users  # [0, n_items) -> [n_users, n_users+n_items)
    cf_ = cf.copy()
    cf_[:, 0], cf_[:, 1] = cf[:, 1], cf[:, 0]  # user->item, item->user

    # diag = np.array([[i, i] for i in range(n_users+n_items)])
    # cf_ = np.concatenate([cf, cf_, diag], axis=0)  # [[0, R], [R^T, 0]] + I
    cf_ = np.concatenate([cf, cf_], axis=0)  # [[0, R], [R^T, 0]]

    vals = [1.] * len(cf_)
    mat = sp.coo_matrix((vals, (cf_[:, 0], cf_[:, 1])), shape=(n_users+n_items, n_users+n_items))
    return _bi_norm_lap(mat)

def build_norm_adj(data_cf, norm_type='sym', split='sub'):
    """
    统一的邻接矩阵构建函数
    
    :param data_cf: 交互数据 [user_id, item_id]
    :param norm_type: 归一化类型，可选 'raw'(无), 'left'(D^-1), 'sym'(D^-1/2)
    :param split: 矩阵形状，可选 'sub'(返回 N x M 子块), 'full'(返回 N+M x N+M 全图)
    :return: scipy.sparse.coo_matrix
    """
    rows = data_cf[:, 0]
    cols = data_cf[:, 1]
    
    # 根据是否需要全图，决定初始矩阵的 shape
    if split == 'full':
        cols = cols + n_users  # 映射到全图坐标
        shape = (n_users + n_items, n_users + n_items)
        # 构建完整的双向边
        R = sp.coo_matrix((np.ones_like(rows, dtype=np.float64), (rows, cols)), shape=shape)
        R_T = sp.coo_matrix((np.ones_like(rows, dtype=np.float64), (cols, rows)), shape=shape)
        adj = R + R_T
    else:
        # 只构建 N x M 的子图
        adj = sp.coo_matrix((np.ones_like(rows, dtype=np.float64), (rows, cols)), shape=(n_users, n_items))

    # --- 开始归一化 ---
    if norm_type == 'raw':
        # 返回未归一化的原始邻接矩阵 (sub时为[N, M], full时为[N+M, N+M])
        return adj.tocoo()
        
    if split == 'sub':
        # 子图模式：需要分别计算行(用户)度和列(物品)度
        rowsum = np.array(adj.sum(1)).flatten()
        colsum = np.array(adj.sum(0)).flatten()
        
        if norm_type == 'left':
            d_u_inv = np.power(rowsum, -1)
            d_u_inv[np.isinf(d_u_inv)] = 0.
            # 返回子图左归一化矩阵: D_U^{-1} * R，形状为 [n_users, n_items]
            return sp.diags(d_u_inv).dot(adj).tocoo()
            
        elif norm_type == 'sym':
            d_u_inv_half = np.power(rowsum, -0.5)
            d_i_inv_half = np.power(colsum, -0.5)
            d_u_inv_half[np.isinf(d_u_inv_half)] = 0.
            d_i_inv_half[np.isinf(d_i_inv_half)] = 0.
            # 返回子图对称归一化矩阵: D_U^{-1/2} * R * D_I^{-1/2}，形状为 [n_users, n_items]
            return sp.diags(d_u_inv_half).dot(adj).dot(sp.diags(d_i_inv_half)).tocoo()
            
    else:
        # 全图模式：只有一个统一的度数矩阵
        rowsum = np.array(adj.sum(1)).flatten()
        
        if norm_type == 'left':
            d_inv = np.power(rowsum, -1)
            d_inv[np.isinf(d_inv)] = 0.
            # 返回全图左归一化矩阵: D^{-1} * A，形状为 [n_users+n_items, n_users+n_items]
            return sp.diags(d_inv).dot(adj).tocoo()
            
        elif norm_type == 'sym':
            d_inv_half = np.power(rowsum, -0.5)
            d_inv_half[np.isinf(d_inv_half)] = 0.
            # 返回全图对称归一化矩阵: D^{-1/2} * A * D^{-1/2}，形状为 [n_users+n_items, n_users+n_items]
            return sp.diags(d_inv_half).dot(adj).dot(sp.diags(d_inv_half)).tocoo()

    raise ValueError(f"Unknown norm_type: {norm_type}")



def build_B(data_cf, mode='sym', a=0.5, b=0.5):
    """构建用户聚合矩阵 B，用户表示为 Z_U = B @ Z_I。

    Parameters
    ----------
    data_cf : numpy.ndarray of int, shape (num_interactions, 2)
        训练交互，每行为 [user_id, item_id]。
        物品 ID 已映射到 [0, n_items)。
    mode : str, optional
        聚合方式，默认为 'sym'。可选值如下：

        - 'sum'：B = R，历史物品直接求和。
        - 'mean'：B = D_U^(-1) R，历史物品均值。
        - 'sqrt'：B = D_U^(-1/2) R，按历史长度的平方根缩放。
        - 'sym'：B = D_U^(-1/2) R D_I^(-1/2)，当前模型的方式。
        - 'degree'：B = D_U^(-a) R D_I^(-b)，自定义度指数。
        - 'weighted_mean'：先按物品度的负 b 次幂降权，再逐行归一化。

    a : float, optional
        用户度指数，默认为 0.5。仅 'degree' 使用，须为有限非负数。
    b : float, optional
        物品度指数，默认为 0.5。'degree' 和 'weighted_mean' 使用，
        须为有限非负数；0 表示不按物品度降权。

    Returns
    -------
    B : scipy.sparse.coo_matrix
        形状为 (n_users, n_items)，无历史用户对应零行。

    Notes
    -----
    使用 statistics() 初始化的全局 n_users、n_items。
    重复交互统一视为一次交互。
    'weighted_mean' 的元素为 R_ui * d_i^(-b) / sum_j(R_uj * d_j^(-b))。

    Examples
    --------
    >>> B = build_B(train_cf, mode='sum')                  # 历史求和
    >>> B = build_B(train_cf, mode='mean')                 # 历史均值
    >>> B = build_B(train_cf, mode='sqrt')                 # 用户度平方根缩放
    >>> B = build_B(train_cf, mode='sym')                  # 当前方案
    >>> B = build_B(train_cf, mode='degree', a=0.5, b=0.3)  # 自定义度指数
    >>> B = build_B(train_cf, mode='weighted_mean', b=0.5)  # 加权均值
    """
    presets = {'sum': (0., 0.), 'mean': (1., 0.), 'sqrt': (0.5, 0.), 'sym': (0.5, 0.5)}
    if mode in presets:
        a, b = presets[mode]
    elif mode == 'weighted_mean':
        a = 0.  # 加权均值通过最终行归一化处理用户尺度，不使用 a
    elif mode != 'degree':
        raise ValueError(f"Unknown B mode: {mode}")
    if not np.isfinite(a) or not np.isfinite(b) or a < 0 or b < 0:
        raise ValueError('B degree exponents must be finite and non-negative')

    # 构建二值交互矩阵 R，确保重复记录不会额外增加历史物品的权重
    rows, cols = data_cf[:, 0], data_cf[:, 1]
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_users, n_items))
    R.data[:] = 1.0
    user_degree, item_degree = np.asarray(R.sum(axis=1)).ravel(), np.asarray(R.sum(axis=0)).ravel()

    # 零度节点权重设为 0，避免负幂产生 inf；非零度节点按指定指数缩放
    user_weight, item_weight = np.zeros(n_users), np.zeros(n_items)
    user_mask, item_mask = user_degree > 0, item_degree > 0
    user_weight[user_mask] = user_degree[user_mask] ** -a
    item_weight[item_mask] = item_degree[item_mask] ** -b
    B = sp.diags(user_weight).dot(R).dot(sp.diags(item_weight))

    if mode == 'weighted_mean':
        # 用降权后的实际权重和归一化，使每个非空用户行的权重和为 1
        rowsum = np.asarray(B.sum(axis=1)).ravel()
        row_inv = np.zeros(n_users)
        mask = rowsum > 0
        row_inv[mask] = 1.0 / rowsum[mask]
        B = sp.diags(row_inv).dot(B)
    return B.tocoo()


def build_S(data_cf, mode='cooccurrence', norm_type=None, gamma=1.0,
            remove_diag=None, top_k=None, threshold=0.0, rho=1.0):
    """构建物品传播矩阵 S，每层传播为 E_next = S @ E。

    Parameters
    ----------
    data_cf : numpy.ndarray of int, shape (num_interactions, 2)
        训练交互 [user_id, item_id]，物品 ID 已映射到 [0, n_items)。
    mode : str, optional
        关联权重方案，默认为 'cooccurrence'。

        - 'cooccurrence'：W = R.T @ R，原始共现次数。
        - 'user_weighted'：W = R.T @ D_U^(-gamma) @ R，共同用户降权。
        - 'cosine'：W = D_I^(-1/2) @ R.T @ R @ D_I^(-1/2)。
        - 'jaccard'：W_ij = c_ij / (d_i + d_j - c_ij)。
        - 'ui'：W = B_sym.T @ B_sym，UI 图两步传播的物品块。

    norm_type : str or None, optional
        权重构建和筛选后的归一化方式。
        None 表示 'ui' 使用 'raw'，其他方案使用 'sym'。

        - 'raw'：S = W，不再归一化。
        - 'left'：S = D_W^(-1) @ W，非空行权重和为 1。
        - 'sym'：S = D_W^(-1/2) @ W @ D_W^(-1/2)。

    gamma : float, optional
        共同用户的度指数，默认为 1.0，仅 'user_weighted' 使用。
        须为有限非负数；0 表示原始共现，1 表示贡献为 1 / d_u。
    remove_diag : bool or None, optional
        是否去掉自然对角线。None 表示 'ui' 保留，其他方案去掉。
        保留对角线与最终层融合是否包含第 0 层是两个独立设置。
    top_k : int or None, optional
        每行保留最多 k 个非对角邻居，再以最大值合并双向边。
        对称化后的邻居数可能超过 k；None 表示不筛选邻居数量。
        权重相同时优先保留物品 ID 较小的邻居。
    threshold : float, optional
        删除小于此值的非对角关联权重，默认为 0.0。
        阈值作用于归一化前的 W，须为有限非负数。
    rho : float, optional
        最后返回 (1 - rho) * I + rho * S，默认为 1.0。
        范围为 [0, 1]；小于 1 时每层显式保留自身信息。

    Returns
    -------
    S : scipy.sparse.coo_matrix
        形状为 (n_items, n_items) 的物品传播矩阵。

    Notes
    -----
    使用 statistics() 初始化的全局 n_users、n_items，重复交互视为一次。
    顺序为：关联权重、对角线处理、阈值、Top-k、对称化、归一化、残差。
    'ui' 仅在不筛选、不额外归一化、保留对角线且 rho=1 时等于 B_sym.T @ B_sym。
    Top-k 在投影后执行，不能避免构造 R.T @ R 时的内存开销。
    直接使用余弦或 Jaccard 相似度传播时，请显式设置 norm_type='raw'。

    Examples
    --------
    >>> S = build_S(train_cf)  # 当前方案：共现、去对角、对称归一化
    >>> S = build_S(train_cf, mode='user_weighted', gamma=1.0)
    >>> S = build_S(train_cf, mode='cosine', norm_type='raw')
    >>> S = build_S(train_cf, mode='jaccard', norm_type='left')
    >>> S = build_S(train_cf, mode='ui')  # 精确的 UI 两步传播物品块
    >>> S = build_S(train_cf, top_k=50, threshold=2.0, rho=0.8)
    """
    if mode not in ('cooccurrence', 'user_weighted', 'cosine', 'jaccard', 'ui'):
        raise ValueError(f"Unknown S mode: {mode}")
    if norm_type is None:
        norm_type = 'raw' if mode == 'ui' else 'sym'
    if norm_type not in ('raw', 'left', 'sym'):
        raise ValueError(f"Unknown S norm_type: {norm_type}")
    if remove_diag is None:
        remove_diag = mode != 'ui'
    if not isinstance(remove_diag, (bool, np.bool_)):
        raise ValueError('remove_diag must be bool or None')
    if top_k is not None and (isinstance(top_k, (bool, np.bool_)) or not isinstance(top_k, (int, np.integer)) or top_k < 1):
        raise ValueError('top_k must be a positive integer or None')
    if not np.isfinite(threshold) or threshold < 0 or not np.isfinite(rho) or not 0 <= rho <= 1:
        raise ValueError('threshold must be finite and non-negative; rho must be in [0, 1]')
    if mode == 'user_weighted' and (not np.isfinite(gamma) or gamma < 0):
        raise ValueError('gamma must be finite and non-negative')

    # 打印解析默认值后的实际构图配置；ui 的用户度指数固定为 1，其他无关模式不使用 gamma
    print(f"[build_S] mode={mode}, norm_type={norm_type}, \
          gamma={gamma if mode == 'user_weighted' else '1.0 (fixed)' if mode == 'ui' else 'N/A'}, \
          remove_diag={remove_diag}, top_k={top_k}, threshold={threshold}, rho={rho}")

    # 构建二值 R，并分别计算原始 UI 图上的用户度、物品度
    rows, cols = data_cf[:, 0], data_cf[:, 1]
    R = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_users, n_items))
    R.data[:] = 1.0
    user_degree, item_degree = np.asarray(R.sum(axis=1)).ravel(), np.asarray(R.sum(axis=0)).ravel()

    # 共同用户降权；ui 使用 gamma=1，并在下方按原始物品度缩放
    if mode in ('user_weighted', 'ui'):
        user_weight = np.zeros(n_users)
        mask = user_degree > 0
        user_weight[mask] = user_degree[mask] ** (-1.0 if mode == 'ui' else -gamma)
        W = (R.T @ sp.diags(user_weight) @ R).tocsr()
    else:
        W = (R.T @ R).tocsr()

    if mode in ('cosine', 'ui'):
        item_weight = np.zeros(n_items)
        mask = item_degree > 0
        item_weight[mask] = item_degree[mask] ** -0.5
        D = sp.diags(item_weight)
        W = (D @ W @ D).tocsr()
    elif mode == 'jaccard':
        W = W.tocoo()
        W.data /= item_degree[W.row] + item_degree[W.col] - W.data
        W = W.tocsr()

    # 暂存自然对角线，使邻居筛选只作用于其他物品
    diagonal = np.zeros(n_items) if remove_diag else W.diagonal().copy()
    W.setdiag(0)
    W.eliminate_zeros()
    W.data[W.data < threshold] = 0.
    W.eliminate_zeros()
    if top_k is not None:
        for i in range(n_items):
            start, end = W.indptr[i:i + 2]
            if end - start > top_k:
                order = np.lexsort((W.indices[start:end], -W.data[start:end]))
                W.data[start + order[top_k:]] = 0.
        W.eliminate_zeros()
        W = W.maximum(W.T).tocsr()  # 任一方向保留即保留双向边，取较大权重
    W.setdiag(diagonal)
    W.eliminate_zeros()

    # 根据筛选后 W 的度归一化；它与原始物品度 D_I 不同
    if norm_type != 'raw':
        degree = np.asarray(W.sum(axis=1)).ravel()
        weight = np.zeros(n_items)
        mask = degree > 0
        weight[mask] = degree[mask] ** (-1.0 if norm_type == 'left' else -0.5)
        D = sp.diags(weight)
        W = D @ W if norm_type == 'left' else D @ W @ D
    if rho != 1.0:
        W = (1.0 - rho) * sp.eye(n_items, format='csr') + rho * W
    W.eliminate_zeros()
    return W.tocoo()

# =============================================

def build_norm_adj2(data_cf, norm_type='sym', split='sub'):
    """
    第二版构图函数, 此次加入了纯粹的II图选项
    统一的邻接矩阵构建函数

    :param data_cf: 交互数据 [user_id, item_id]
    :param norm_type:
        'raw'  : 不归一化
        'left' : D^-1 A
        'sym'  : D^-1/2 A D^-1/2

    :param split:
        'sub'  : 返回 user-item 子图, shape = [n_users, n_items]
        'full' : 返回 user-item 双向全图, shape = [n_users+n_items, n_users+n_items]
        'ii'   : 返回 item-item 共现图, shape = [n_items, n_items]

    :return: scipy.sparse.coo_matrix
    """

    rows, cols = data_cf[:, 0], data_cf[:, 1]

    # ==================== Item-Item 图 ====================
    if split == 'ii':
        # 构建二值 User-Item 交互矩阵 R，形状为 [n_users, n_items]
        R = sp.csr_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_users, n_items))
        R.data[:] = 1.0  # 如果存在重复交互，统一视为一次交互

        # 构建 Item-Item 共现矩阵 C = R.T @ R
        # C[i, j] 表示同时与 item i 和 item j 交互过的用户数量
        adj = (R.T @ R).tocsr()

        # 去掉 item 自己和自己的共现
        adj.setdiag(0)
        adj.eliminate_zeros()

        # 不进行归一化，直接返回原始 Item-Item 共现矩阵
        if norm_type == 'raw': 
            return adj.tocoo()

        # Item-Item 图中的度：每个 item 与其他 item 共现边权之和
        rowsum = np.asarray(adj.sum(axis=1)).ravel()

        if norm_type == 'left':
            # 左归一化：D^-1 C
            d_inv = np.zeros_like(rowsum, dtype=np.float64)
            mask = rowsum > 0
            d_inv[mask] = rowsum[mask] ** -1
            return sp.diags(d_inv).dot(adj).tocoo()

        if norm_type == 'sym':
            # 对称归一化：D^-1/2 C D^-1/2
            d_inv_half = np.zeros_like(rowsum, dtype=np.float64)
            mask = rowsum > 0
            d_inv_half[mask] = rowsum[mask] ** -0.5
            D = sp.diags(d_inv_half)
            return D.dot(adj).dot(D).tocoo()

    # ==================== User-Item 全图 ====================
    elif split == 'full':
        # 将 item id 映射到 [n_users, n_users+n_items) 的全图坐标
        cols_full = cols + n_users
        shape = (n_users + n_items, n_users + n_items)

        # 构建 User -> Item 和 Item -> User 双向边
        R = sp.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols_full)), shape=shape)
        R_T = sp.coo_matrix((np.ones(len(rows), dtype=np.float64), (cols_full, rows)), shape=shape)
        adj = R + R_T

    # ==================== User-Item 子图 ====================
    elif split == 'sub':
        # 只构建 User-Item 二部图子矩阵 R，形状为 [n_users, n_items]
        adj = sp.coo_matrix((np.ones(len(rows), dtype=np.float64), (rows, cols)), shape=(n_users, n_items))

    else:
        raise ValueError(f"Unknown split type: {split}")

    # 不归一化，直接返回原始邻接矩阵
    if norm_type == 'raw': return adj.tocoo()

    # ==================== User-Item 子图归一化 ====================
    if split == 'sub':
        # 分别计算用户度和物品度
        rowsum, colsum = np.asarray(adj.sum(axis=1)).ravel(), np.asarray(adj.sum(axis=0)).ravel()

        if norm_type == 'left':
            # 左归一化：D_U^-1 R
            d_u_inv = np.zeros_like(rowsum, dtype=np.float64)
            mask = rowsum > 0
            d_u_inv[mask] = rowsum[mask] ** -1
            return sp.diags(d_u_inv).dot(adj).tocoo()

        if norm_type == 'sym':
            # 对称归一化：D_U^-1/2 R D_I^-1/2
            d_u_inv_half, d_i_inv_half = np.zeros_like(rowsum, dtype=np.float64), np.zeros_like(colsum, dtype=np.float64)
            user_mask, item_mask = rowsum > 0, colsum > 0
            d_u_inv_half[user_mask], d_i_inv_half[item_mask] = rowsum[user_mask] ** -0.5, colsum[item_mask] ** -0.5
            return sp.diags(d_u_inv_half).dot(adj).dot(sp.diags(d_i_inv_half)).tocoo()

    # ==================== User-Item 全图归一化 ====================
    if split == 'full':
        # 全图中统一计算每个节点的度
        rowsum = np.asarray(adj.sum(axis=1)).ravel()

        if norm_type == 'left':
            # 左归一化：D^-1 A
            d_inv = np.zeros_like(rowsum, dtype=np.float64)
            mask = rowsum > 0
            d_inv[mask] = rowsum[mask] ** -1
            return sp.diags(d_inv).dot(adj).tocoo()

        if norm_type == 'sym':
            # 对称归一化：D^-1/2 A D^-1/2
            d_inv_half = np.zeros_like(rowsum, dtype=np.float64)
            mask = rowsum > 0
            d_inv_half[mask] = rowsum[mask] ** -0.5
            D = sp.diags(d_inv_half)
            return D.dot(adj).dot(D).tocoo()

    raise ValueError(f"Unknown norm_type: {norm_type}")


def load_data(model_args):
    global args, dataset
    args = model_args
    dataset = args.dataset
    directory = args.data_path + dataset + '/'

    if dataset in ['yelp2018','movielens','gowalla'] :
        read_cf = read_cf_yelp2018
    else:
        read_cf = read_cf_amazon

    print('reading train and test user-item set ...')
    train_cf = read_cf(directory + 'train.txt')
    test_cf = read_cf(directory + 'test.txt')
    if args.dataset not in ['yelp2018','gowalla']:
        valid_cf = read_cf(directory + 'valid.txt')
    else:
        valid_cf = test_cf
    train_sp_mat, valid_sp_mat, test_sp_mat, valid_pre, test_pre = statistics(train_cf, valid_cf, test_cf)

    print('building the adj mat ...')
    """define model"""
    if args.gnn == 'lightgcn':
        norm_mat = build_sparse_graph(train_cf)
        si_sub_norm_mat = None
    elif args.gnn in ('simgcl', 'xsimgcl', 'sgl', 'xsgl', 'recdcl', 'xrecdcl',
                      'xlightgcn', 'ahns', 'xahns', 'directau', 'xdirectau', 'graphau', 'xgraphau'):
        norm_mat = build_sparse_graph(train_cf)
        si_sub_norm_mat = (build_B(train_cf, mode=args.b_mode, a=args.b_a, b=args.b_b)
                           if args.gnn.startswith('x') else None)
    elif args.gnn == 'igcn':
        # ===== 本次新增：接入 parser 的 B / S 构图参数 BEGIN ===== #
        # auto 转为 None，由 build_S 根据 mode 决定默认行为
        norm_mat = build_S(
            train_cf, mode=args.s_mode,
            norm_type=None if args.s_norm == 'auto' else args.s_norm,
            gamma=args.s_gamma,
            remove_diag=None if args.s_remove_diag == 'auto' else args.s_remove_diag == 'true',
            top_k=args.s_top_k, threshold=args.s_threshold, rho=args.s_rho)
        
        si_sub_norm_mat = build_B(train_cf, mode=args.b_mode, a=args.b_a, b=args.b_b)
        # ===== 本次新增：接入 parser 的 B / S 构图参数 END ===== #
    else:
        raise NotImplementedError("unknown gnn type: " + args.gnn)

    n_params = {
        'n_users': int(n_users),
        'n_items': int(n_items),
    }
    user_dict = {
        'train_user_set': train_user_set,
        'valid_user_set': valid_user_set if args.dataset not in ['yelp2018','gowalla'] else None,
        'test_user_set': test_user_set,
    }
    sp_matrix = {
        'train_sp_mat': train_sp_mat,
        'valid_sp_mat': valid_sp_mat,
        'test_sp_mat': test_sp_mat
    }
    print('loading over ...')
    return train_cf, user_dict, sp_matrix, n_params, norm_mat, si_sub_norm_mat, valid_pre, test_pre, None
