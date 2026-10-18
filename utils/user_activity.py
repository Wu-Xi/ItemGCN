"""Fixed user quartiles, per-user ranking metrics, and paired descriptive gains."""
import numpy as np

METRICS = ('recall', 'ndcg', 'hit_ratio')


def partition_users(user_ids, train_user_set):
    """Four near-equal groups sorted by (distinct TRAIN degree, numeric user ID)."""
    user_ids = np.asarray(user_ids, dtype=np.int64)
    if len(user_ids) < 4 or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError('Require at least four unique test users')
    degrees = np.array([len(set(train_user_set.get(int(u), ()))) for u in user_ids])
    order = np.lexsort((user_ids, degrees))
    groups = np.empty(len(user_ids), dtype=np.int64)
    description = []
    for g, positions in enumerate(np.array_split(order, 4), 1):
        groups[positions] = g
        values = degrees[positions]
        description.append(dict(group=g, users=len(positions), degree_min=int(values.min()),
                                degree_max=int(values.max()), degree_mean=float(values.mean())))
    return groups, degrees, description


def metrics_from_hits(hits, positive_counts, ks):
    """User macro metrics, including the same IDCG convention as utils.evaluate."""
    hits = np.asarray(hits, dtype=bool)
    counts = np.asarray(positive_counts, dtype=np.int64)
    if hits.ndim != 2 or counts.shape != (len(hits),) or np.any(counts <= 0):
        raise ValueError('Every evaluated user must have test positives')
    if not ks or min(ks) < 1 or max(ks) > hits.shape[1]:
        raise ValueError('Invalid ranking cutoffs')
    cutoffs = np.asarray(ks, dtype=np.int64)
    cumulative_hits = np.cumsum(hits, axis=1)[:, cutoffs - 1]
    discount = 1.0 / np.log2(np.arange(hits.shape[1]) + 2.0)
    dcg = np.cumsum(hits * discount, axis=1)[:, cutoffs - 1]
    idcg = np.cumsum(discount)[np.minimum(counts[:, None], cutoffs[None, :]) - 1]
    return dict(recall=cumulative_hits / counts[:, None], ndcg=dcg / idcg,
                hit_ratio=(cumulative_hits > 0).astype(float))


def evaluate_users(model, data, ks, batch_size):
    """Use the existing evaluator's score permutation/topk to match its tie behavior.

    Returned columns are metrics, not real item IDs. This routine is user-only.
    """
    import torch
    _, users, matrices, sizes, _, _, _, test_pre, _ = data
    test_users = list(users['test_user_set'])
    if batch_size < 1 or max(ks) > sizes['n_items']:
        raise ValueError('Invalid batch size or K exceeds catalog')
    swap_indices, reverse_indices, positive_counts = test_pre
    if any(len(set(users['test_user_set'][u])) != n for u, n in zip(test_users, positive_counts)):
        raise ValueError('Duplicate test positives: resolve metric convention before evaluating')
    train = matrices['train_sp_mat']
    batches = []
    model.eval()
    with torch.no_grad():
        user_embeddings, item_embeddings = model.generate()
        device = user_embeddings.device
        for start in range(0, len(test_users), batch_size):
            ids = test_users[start:start + batch_size]
            scores = model.rating(user_embeddings[torch.tensor(ids, device=device)], item_embeddings)
            if not torch.isfinite(scores).all().item():
                raise ValueError('Non-finite model scores')
            rows, cols = train[ids].nonzero()
            scores[torch.as_tensor(rows, device=device, dtype=torch.long),
                   torch.as_tensor(cols, device=device, dtype=torch.long)] = -float('inf')
            swaps = swap_indices[ids]
            swap_rows = torch.cat([torch.full_like(s, i) for i, s in enumerate(swaps)]).long().to(device)
            after = torch.cat(list(swaps)).long().to(device)
            before = torch.cat(list(reverse_indices[ids])).long().to(device)
            scores[swap_rows, after] = scores[swap_rows, before]
            values, indices = torch.topk(scores, max(ks), dim=-1)
            counts = np.asarray(positive_counts[start:start + len(ids)])
            # Masked catalog entries must never count as hits, even if fewer than K remain.
            hits = ((indices.cpu().numpy() < counts[:, None]) & torch.isfinite(values).cpu().numpy())
            batches.append(hits)
            if start == 0 or start + batch_size >= len(test_users) or (start // batch_size) % 50 == 0:
                print(f'  ranked {min(start + batch_size, len(test_users))}/{len(test_users)} users', flush=True)
    metrics = metrics_from_hits(np.concatenate(batches), positive_counts, ks)
    return np.asarray(test_users, dtype=np.int64), metrics


def aggregate(metrics, groups):
    return {name: {str(g): values[groups == g].mean(axis=0).tolist() for g in range(1, 5)}
            for name, values in metrics.items()}


def paired_gains(original, item_only, groups, ks):
    """Weighted group contributions sum to the overall macro-metric difference."""
    rows = []
    for metric in METRICS:
        for column, k in enumerate(ks):
            base, improved = original[metric][:, column], item_only[metric][:, column]
            if base.shape != groups.shape or improved.shape != groups.shape:
                raise ValueError('Paired arrays must have the same users')
            for g in range(1, 5):
                mask = groups == g
                a, b = float(base[mask].mean()), float(improved[mask].mean())
                delta = b - a
                rows.append(dict(metric=metric, k=int(k), group=g, users=int(mask.sum()),
                                 original=a, item_only=b, absolute_gain=delta,
                                 relative_gain_pct=(100 * delta / a if a > 0 else None),
                                 overall_gain_contribution=float(mask.mean()) * delta))
    return rows
