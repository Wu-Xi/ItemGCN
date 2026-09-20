import argparse
import math

def str2bool(v):
    if isinstance(v, bool):
        return v
    if str(v).strip().lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    if str(v).strip().lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    raise argparse.ArgumentTypeError(f'期望布尔值，收到: {v}')

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="DINS")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="yelp2018",
                        help="Choose a dataset:[amazon,yelp2018,ali,ifashion,gowalla]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    # ===== train ===== # 
    parser.add_argument("--gnn", nargs="?", default="lightgcn",
                        help="Choose a recommender:[lightgcn, ngcf, mf]")
    parser.add_argument('--epoch', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--test_batch_size', type=int, default=2048, help='batch size in evaluation phase')
    parser.add_argument('--dim', type=int, default=None, help='embedding size')
    parser.add_argument('--l2', type=float, default=None, help='l2 regularization weight, 1e-5 for NGCF')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument("--mess_dropout", type=str2bool, default=False, help="consider mess dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float, default=0.1, help="ratio of mess dropout")
    parser.add_argument("--edge_dropout", type=str2bool, default=False, help="consider edge dropout or not")
    parser.add_argument("--edge_dropout_rate", type=float, default=0.1, help="ratio of edge sampling")
    parser.add_argument("--batch_test_flag", type=str2bool, default=True, help="use gpu or not")

    # ===== COMMON ===== # 
    parser.add_argument("--ns", type=str, default='rns', help="LightGCN/XLightGCN: rns, mixgcf, stabcf; AHNS uses its own sampler")
    parser.add_argument("--n_negs", type=int, default=64, help="number of candidate negative")
    parser.add_argument("--pool", type=str, default='mean', help="[concat, mean, sum, final]")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha")
    parser.add_argument("--embedding0", type=str2bool, default=True, help="Contain initialisation embedding or not")
    parser.add_argument("--norm_type", type=str, default='sym', help="[raw, left, sym]")
    parser.add_argument("--split", type=str, default='sub', help="[sub, full]")
    parser.add_argument("--abl", type=str, default='full', help="ablation study [WP,WT,WA,WG,full]")

    # ===== 本次新增：StabCF2 的 B / S 构图参数 BEGIN ===== #
    graph = parser.add_argument_group('StabCF2 B / S graph construction')
    # B = D_U^(-a) R D_I^(-b)；预设模式会使用其固定指数
    graph.add_argument('--b_mode', choices=['sum', 'mean', 'sqrt', 'sym', 'degree', 'weighted_mean'],
                       default='sym', help='B 聚合方式，默认 sym')
    graph.add_argument('--b_a', type=float, default=0.5, help='B 用户度指数，仅 degree 使用')
    graph.add_argument('--b_b', type=float, default=0.5, help='B 物品度指数，degree / weighted_mean 使用')
    
    # S：关联权重 -> 对角线 -> 筛选 -> 归一化 -> 残差
    graph.add_argument('--s_mode', choices=['cooccurrence', 'user_weighted', 'cosine', 'jaccard', 'ui'],
                       default='cooccurrence', help='S 关联权重方案，默认 cooccurrence')
    graph.add_argument('--s_norm', choices=['auto', 'raw', 'left', 'sym'], default='auto',
                       help='auto：ui 使用 raw，其他模式使用 sym')
    graph.add_argument('--s_gamma', type=float, default=1.0,
                       help='共同用户度指数，仅 user_weighted 使用；独立于 DENS 的 --gamma')
    graph.add_argument('--s_remove_diag', choices=['auto', 'true', 'false'], default='auto', type=str.lower,
                       help='auto：ui 保留对角线，其他模式去掉；true 去掉，false 保留')
    graph.add_argument('--s_top_k', type=int, default=None,
                       help='每行选 k 个非对角邻居再对称化，最终邻居可能超过 k；省略表示不限')
    graph.add_argument('--s_threshold', type=float, default=0.0,
                       help='删除归一化前权重低于此值的非对角边，默认 0')
    graph.add_argument('--s_rho', type=float, default=1.0,
                       help='最终传播为 (1-rho)I + rho*S，范围 [0,1]，默认 1')
    # ===== 本次新增：StabCF2 的 B / S 构图参数 END ===== #
    
    # ===== model test ===== # 
    parser.add_argument("--cuda", type=str2bool, default=True, help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument('--Ks', nargs='?', default='[10, 20, 50]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')

    parser.add_argument("--context_hops", type=int, default=None, help="hop")

    # ===== save model ===== #
    parser.add_argument("--save", type=str2bool, default=False, help="save model or not")
    parser.add_argument("--out_dir", type=str, default="./weights/", help="output directory for model")

    parser.add_argument('--encoder', choices=['mf', 'lightgcn'], default='lightgcn')
    parser.add_argument('--cl_rate', type=float, default=None)
    parser.add_argument('--eps', type=float, default=0.1)
    parser.add_argument('--ssl_temp', type=float, default=0.2)
    parser.add_argument('--drop_rate', type=float, default=0.1)
    parser.add_argument('--aug_type', type=int, choices=[0, 1, 2], default=1)
    parser.add_argument('--a', type=float, default=1.)
    parser.add_argument('--polyc', type=float, default=1e-7)
    parser.add_argument('--degree', type=int, default=4)
    parser.add_argument('--poly_coeff', type=float, default=.2)
    parser.add_argument('--bt_coeff', type=float, default=.01)
    parser.add_argument('--all_bt_coeff', type=float, default=1.)
    parser.add_argument('--mom_coeff', type=float, default=10.)
    parser.add_argument('--momentum', type=float, default=.3)
    parser.add_argument('--simi', choices=['ip', 'cos', 'ed'], default='ip')
    parser.add_argument('--p', type=float, default=-0.5, help='AHNS exponent parameter')
    parser.add_argument('--beta', type=float, default=0.5, help='AHNS score scale')
    parser.add_argument('--gamma', type=float, default=None, help='AU uniformity weight')
    parser.add_argument('--graphau_layers', type=int, default=4)
    parser.add_argument('--decaying_base', type=float, default=1.4)
    parser.add_argument('--seed', type=int, default=2025, help='Single manually selected seed')
    parser.add_argument('--weight_decay', type=float, default=0., help='Adam weight decay; distinct from model l2')
    parser.add_argument('--run_dir', type=str, default=None, help='Optional per-run machine-readable records')
    parser.add_argument('--window_length', type=int, default=5, help='StabCF historical positives per interaction')
    args = parser.parse_args(argv)
    aliases = {'rns': ('lightgcn', 'rns'), 'mixgcf': ('lightgcn', 'mixgcf'),
               'stabcf': ('lightgcn', 'stabcf'), 'xrns': ('xlightgcn', 'rns'),
               'xmixgcf': ('xlightgcn', 'mixgcf'), 'xstabcf': ('xlightgcn', 'stabcf')}
    if args.gnn in aliases:
        args.gnn, args.ns = aliases[args.gnn]
    if args.gnn in ('lightgcn', 'xlightgcn'):
        if args.ns not in ('rns', 'mixgcf', 'stabcf'):
            parser.error('CF sampling must be rns, mixgcf or stabcf')
        if args.n_negs < 1 or (args.ns == 'stabcf' and args.window_length < 1):
            parser.error('n_negs and StabCF window_length must be positive')
    family = args.gnn[1:] if args.gnn.startswith('x') else args.gnn
    if args.dim is None:
        args.dim = 2048 if family == 'recdcl' else 64
    if args.context_hops is None:
        args.context_hops = 3
    if args.l2 is None:
        args.l2 = 1e-3
    if args.cl_rate is None:
        args.cl_rate = .5 if family == 'simgcl' else .1
    if family in ('simgcl', 'sgl', 'recdcl'):
        if args.dim < 1 or args.batch_size < 1 or args.context_hops < 0:
            parser.error('dim/batch_size must be positive; context_hops must be nonnegative')
        if family in ('simgcl', 'sgl'):
            if args.context_hops < 1:
                parser.error('SimGCL/SGL require at least one propagation layer')
            if not math.isfinite(args.ssl_temp) or args.ssl_temp <= 0:
                parser.error('ssl_temp must be finite and positive')
            if not 0 <= args.drop_rate <= 1:
                parser.error('drop_rate must be in [0, 1]')
        for name in ('cl_rate', 'eps', 'l2', 'a', 'polyc', 'poly_coeff', 'bt_coeff', 'all_bt_coeff', 'mom_coeff'):
            if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
                parser.error(name + ' must be finite and nonnegative')
        if family == 'recdcl':
            if not 0 <= args.momentum <= 1 or args.degree <= 0 or args.degree % 2:
                parser.error('momentum must be in [0, 1]; degree must be positive and even')
            if args.poly_coeff and args.polyc <= 0:
                parser.error('polyc must be positive when polynomial loss is enabled')
            if args.all_bt_coeff + args.poly_coeff + args.mom_coeff == 0:
                parser.error('At least one RecDCL loss must be enabled')
    return args
