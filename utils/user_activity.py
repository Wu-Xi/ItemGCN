"""User activity grouping, per-user ranking metrics, and paired descriptive gains."""
import numpy as np

METRICS = ('recall', 'ndcg', 'hit_ratio')
GROUP_METHODS = ('equal_freq', 'equal_width', 'fixed')
DEFAULT_THRESHOLDS = (5, 20, 50)


def partition_users(user_ids, train_user_set, method='equal_freq', thresholds=DEFAULT_THRESHOLDS):
    """Partition test users into G1..G4 by distinct TRAIN interactions.

    equal_freq: near-equal group sizes (original behavior; ties split by user ID).
    equal_width: four equal-width integer degree ranges between min and max.
    fixed: four degree ranges <=t1, (t1,t2], (t2,t3], >t3.

    Empty groups are allowed for equal_width/fixed and get None statistics.
    Returns groups, degrees, and four group-statistic dictionaries.
    """
    user_ids = np.asarray(user_ids, dtype=np.int64)
    if len(user_ids) < 4 or len(np.unique(user_ids)) != len(user_ids):
        raise ValueError('Require at least four unique test users')
    if method not in GROUP_METHODS:
        raise ValueError(f'Unknown group method: {method}; choose from {GROUP_METHODS}')
    degrees = np.array([len(set(train_user_set.get(int(u), ()))) for u in user_ids], dtype=np.int64)

    if method == 'equal_freq':
        order = np.lexsort((user_ids, degrees))
        groups = np.empty(len(user_ids), dtype=np.int64)
        for g, positions in enumerate(np.array_split(order, 4), 1):
            groups[positions] = g
        bounds = None
    else:
        if method == 'fixed':
            if len(thresholds) != 3 or any(isinstance(t, (bool, np.bool_)) or
                    not isinstance(t, (int, np.integer)) or t < 0 for t in thresholds) or not all(
                    a < b for a, b in zip(thresholds, thresholds[1:])):
                raise ValueError('Fixed grouping requires three nonnegative, strictly increasing integer thresholds')
            bounds = [int(t) for t in thresholds]
        else:
            minimum, maximum = int(degrees.min()), int(degrees.max())
            span = maximum - minimum + 1
            bounds = [minimum + (j * span + 3) // 4 - 1 for j in (1, 2, 3)]
        # side='left': degree exactly on a cutoff belongs to the lower-numbered group.
        groups = np.searchsorted(bounds, degrees, side='left').astype(np.int64) + 1

    description = []
    for g in range(1, 5):
        values = degrees[groups == g]
        if bounds is None:
            label = f'{int(values.min())}–{int(values.max())}'
        elif method == 'fixed':
            label = (f'≤{bounds[0]}' if g == 1 else f'≥{bounds[2] + 1}' if g == 4
                     else f'{bounds[g - 2] + 1}–{bounds[g - 1]}')
        else:
            lower = int(degrees.min()) if g == 1 else bounds[g - 2] + 1
            upper = int(degrees.max()) if g == 4 else bounds[g - 1]
            label = f'{lower}–{upper}' if lower <= upper else '∅'
        description.append(dict(group=g, users=int(len(values)), degree_min=int(values.min()) if len(values) else None,
                                degree_max=int(values.max()) if len(values) else None,
                                degree_mean=float(values.mean()) if len(values) else None,
                                degree_range=label))
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
    """Empty groups have null-valued metric columns (JSON-safe)."""
    return {name: {str(g): values[groups == g].mean(axis=0).tolist() if np.any(groups == g)
                   else [None] * values.shape[1] for g in range(1, 5)} for name, values in metrics.items()}


def paired_gains(original, item_only, groups, ks):
    """Weighted group contributions sum to the overall macro-metric difference.

    For empty groups, gains are undefined (None), and their contribution is zero.
    """
    rows = []
    for metric in METRICS:
        for column, k in enumerate(ks):
            base, improved = original[metric][:, column], item_only[metric][:, column]
            if base.shape != groups.shape or improved.shape != groups.shape:
                raise ValueError('Paired arrays must have the same users')
            for g in range(1, 5):
                mask = groups == g
                if not mask.any():
                    rows.append(dict(metric=metric, k=int(k), group=g, users=0,
                                     original=None, item_only=None, absolute_gain=None,
                                     relative_gain_pct=None, overall_gain_contribution=0.0))
                    continue
                a, b = float(base[mask].mean()), float(improved[mask].mean())
                delta = b - a
                rows.append(dict(metric=metric, k=int(k), group=g, users=int(mask.sum()),
                                 original=a, item_only=b, absolute_gain=delta,
                                 relative_gain_pct=(100 * delta / a if a > 0 else None),
                                 overall_gain_contribution=float(mask.mean()) * delta))
    return rows
