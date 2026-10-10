import os
import random
import torch
import numpy as np
from time import time
from prettytable import PrettyTable
from utils.parser import parse_args
from utils.evaluate import test
from utils.helper import early_stopping, get_local_time
from utils.data_loader import load_data
from copy import deepcopy
import pdb
from utils.experiment_log import ExperimentLog
from utils.model_factory import build_model
from utils.checkpoint import BestCheckpoint

def prepare_train_sets(histories, catalog_size):
    if catalog_size <= 0:
        raise ValueError('Negative sampling requires n_items > 0')
    train_sets = {user: set(items) for user, items in histories.items()}
    for user, items in train_sets.items():
        if len(items) >= catalog_size:
            raise ValueError('User {} has no unobserved item for negative sampling'.format(user))
    return train_sets

def get_feed_dictv2(train_entity_pairs, train_pos_set, start, end, n_negs=1, *, train_sets):
    entity_pairs = train_entity_pairs[start:end]
    users = entity_pairs[:, 0].tolist()
    negatives = np.random.randint(0, n_items, size=(len(users), n_negs), dtype=np.int64)
    for row, user in enumerate(users):
        positives = train_sets[user]
        for col in range(n_negs):
            while negatives[row, col] in positives:
                negatives[row, col] = np.random.randint(n_items)

    feed_dict = {
        'users': entity_pairs[:, 0].to(device),
        'pos_items': entity_pairs[:, 1].to(device),
        'neg_items': torch.from_numpy(negatives).to(device)}
    return feed_dict

def get_feed_dictv3(train_entity_pairs, train_pos_set, start, end, n_negs=1, *, train_sets):
    entity_pairs = train_entity_pairs[start:end]
    users = entity_pairs[:, 0].tolist()
    negatives = np.random.randint(0, n_items, size=(len(users), n_negs), dtype=np.int64)
    other_user = np.random.randint(0, n_users, size=(len(users), n_negs), dtype=np.int64)
    for row, user in enumerate(users):
        positives = train_sets[user]
        for col in range(n_negs):
            while negatives[row, col] in positives:
                negatives[row, col] = np.random.randint(n_items)
            while other_user[row, col] == user:
                other_user[row, col] = np.random.randint(n_users)

    feed_dict = {
        'users': entity_pairs[:, 0].to(device),
        'pos_items': entity_pairs[:, 1].to(device),
        'neg_items': torch.from_numpy(negatives).to(device),
        'other_users': torch.from_numpy(other_user).to(device),
        }
    if args.ns == 'stabcf':
        # Keep StabCF's original historical-positive sampling rules.
        observed_pos_list = []
        for user in users:
            all_pos_list = train_pos_set[user]
            if len(all_pos_list) > args.window_length:
                observed_pos_list.append(random.sample(all_pos_list, k=args.window_length))
            else:
                observed_pos_list.append(random.choices(all_pos_list, k=args.window_length))
        observed = np.asarray(observed_pos_list, dtype=np.int64).reshape(len(users), args.window_length)
        feed_dict['observed_pos_items'] = torch.from_numpy(observed).to(device)
    return feed_dict

def get_feed_dictv4(train_entity_pairs, train_pos_set, train_sets, start, end, n_negs=1, requires_negative_sampling=True, requires_history=False):
    entity_pairs = train_entity_pairs[start:end]
    feed_dict = {
        'users': entity_pairs[:, 0].to(device),
        'pos_items': entity_pairs[:, 1].to(device)}
    if not requires_negative_sampling:
        return feed_dict
    
    # negative sampling processing
    users = entity_pairs[:, 0].tolist()
    negatives = np.random.randint(0, n_items, size=(len(users), n_negs), dtype=np.int64)
    for row, user in enumerate(users):
        positives = train_sets[user]
        for col in range(n_negs):
            while negatives[row, col] in positives:
                negatives[row, col] = np.random.randint(n_items)
    feed_dict['neg_items'] = torch.tensor(negatives, dtype=torch.long, device=device)
    if requires_history:
        observed = [random.sample(train_pos_set[u], k=args.window_length)
                    if len(train_pos_set[u]) > args.window_length
                    else random.choices(train_pos_set[u], k=args.window_length) for u in users]
        feed_dict['observed_pos_items'] = torch.as_tensor(
            np.asarray(observed, dtype=np.int64), device=device)

    
    return feed_dict

if __name__ == '__main__':
    """fix the random seed"""
    global args, device, n_items, n_users
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    """read args"""
    print(args)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device("cuda:0") if args.cuda else torch.device("cpu")
    
    experiment = ExperimentLog(args)

    """build dataset"""
    train_cf, user_dict, sp_matrix, n_params, norm_mat, si_norm_mat, valid_pre, test_pre, item_group_idx = load_data(args)
    train_cf = torch.as_tensor(np.asarray(train_cf), dtype=torch.long)

    n_items = n_params['n_items']
    n_users = n_params['n_users']

    """define model"""
    model = build_model(n_params, args, norm_mat, si_norm_mat, sp_matrix['train_sp_mat']).to(device)
    checkpoint = BestCheckpoint(args, n_params) if args.save else None

    requires_negatives = getattr(model, 'requires_negative_sampling', True)
    negative_count = getattr(model, 'negative_sample_count', args.n_negs)
    if args.gnn in ('lightgcn', 'xlightgcn', 'igcn'):
        print('Sampling: {}, candidates={}'.format(args.ns if args.gnn != 'igcn' else 'rns', negative_count))
    train_sets = prepare_train_sets(user_dict['train_user_set'], n_items) if requires_negatives else None
    """define optimizer"""
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    cur_best_pre_0 = -float("inf")
    stopping_step = 0
    should_stop = False

    print(f"{get_local_time()[0]} start training ...")
    best_test_result = None
    for epoch in range(args.epoch):
        # shuffle training data
        train_cf_ = train_cf
        index = np.arange(len(train_cf_))
        np.random.shuffle(index)
        train_cf_ = train_cf_[index]

        """training"""
        model.train()
        loss = torch.zeros((), device=device)
        train_s_t = time()
        if hasattr(model, 'on_train_epoch_start'):
            model.on_train_epoch_start()
        
        for s in range(0, len(train_cf_), args.batch_size):
            batch = get_feed_dictv4(train_cf_ , user_dict['train_user_set'], train_sets, s , s + args.batch_size, negative_count, requires_negatives, getattr(model, 'requires_history', False))
            optimizer.zero_grad(set_to_none=True)
            batch_loss,bpr_loss,reg_loss= model(batch,epoch)
            if not torch.isfinite(batch_loss):
                raise FloatingPointError('Non-finite training loss')
            batch_loss.backward()
            optimizer.step()
            loss += batch_loss.detach()
            
        loss_value = loss.item()
        train_e_t = time()
        if epoch % 1 == 0:
            """testing"""
            train_res = PrettyTable()
            train_res.field_names = ["Epoch", "training time(s)", "tesing time(s)", "Loss", "recall", "ndcg", "precision", "hit_ratio"]
            model.eval()
            test_s_t = time()
            test_ret = test(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='test', evaluation_args=args)
            test_e_t = time()
            test_result = [epoch, int(train_e_t - train_s_t), int(test_e_t - test_s_t), round(loss.item(), 2), *[np.round(test_ret[k], 5).tolist() for k in ('recall', 'ndcg', 'precision', 'hit_ratio')]]
            train_res.add_row(test_result)

            if user_dict['valid_user_set'] is None:
                valid_ret = test_ret
            else:
                test_s_t = time()
                valid_ret = test(model, user_dict, sp_matrix, n_params, valid_pre, test_pre, mode='valid', evaluation_args=args)
                test_e_t = time()
                train_res.add_row(
                    [epoch, int(train_e_t - train_s_t), int(test_e_t - test_s_t), round(loss.item(), 2), *[np.round(valid_ret[k], 5).tolist() for k in ('recall', 'ndcg', 'precision', 'hit_ratio')]])
            print(train_res)

            # *********************************************************
            # early stopping when cur_best_pre_0 is decreasing for 10 successive steps.
            improved = valid_ret['recall'][1] > cur_best_pre_0
            cur_best_pre_0, stopping_step, should_stop = early_stopping(valid_ret['recall'][1], cur_best_pre_0, stopping_step, expected_order='acc', flag_step=10)

            experiment.record(epoch, loss_value, train_e_t - train_s_t,
                              time() - train_e_t, valid_ret, test_ret, improved,
                              'validation' if user_dict['valid_user_set'] is not None else 'test')
            # save weight
            if improved:
                best_test_result = deepcopy(test_result)
                if checkpoint is not None:
                    checkpoint.save(model, epoch, valid_ret, test_ret,
                                    'validation' if user_dict['valid_user_set'] is not None else 'test')
            if should_stop:
                break

        else:
            # logging.info('training loss at epoch %d: %f' % (epoch, round(loss.item(), 2)))
            print('using time %.4fs, training loss at epoch %d: %.4f' % (int(train_e_t - train_s_t), epoch, round(loss.item(), 2)))


    print(f"{get_local_time()[0]} end training ...")
    print('finished at epoch %d, best epoch %d, recall@%d:%.5f' %
          (epoch, best_test_result[0], 20, cur_best_pre_0))
    train_res.clear_rows()
    train_res.add_row(best_test_result)
    print(train_res)

    experiment.finish(epoch, cur_best_pre_0)
