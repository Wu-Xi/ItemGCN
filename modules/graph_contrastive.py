"""Shared QRec-compatible scoring and losses for SimGCL and SGL."""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def sparse_tensor(matrix):
    coo = matrix.tocoo()
    return torch.sparse_coo_tensor(
        torch.from_numpy(np.stack([coo.row, coo.col])).long(),
        torch.as_tensor(coo.data, dtype=torch.float32), coo.shape).coalesce()


def info_nce(first, second, temperature):
    # TensorFlow l2_normalize uses sqrt(max(sum(x**2), 1e-12)).
    first = F.normalize(first, dim=-1, eps=1e-6)
    second = F.normalize(second, dim=-1, eps=1e-6)
    logits = first @ second.T / temperature
    return (torch.logsumexp(logits, dim=1) - logits.diagonal()).sum()


class GraphContrastive(nn.Module):
    requires_negative_sampling = True
    # QRec uses one uniformly drawn negative per interaction, not MixGCF mining.
    negative_sample_count = 1

    def __init__(self, data_config, args_config, adj_mat, si_norm_mat=None):
        super().__init__()
        self.n_users, self.n_items = data_config['n_users'], data_config['n_items']
        self.n_layers = args_config.context_hops
        self.cl_rate = args_config.cl_rate
        self.temperature = args_config.ssl_temp
        self.reg_weight = args_config.l2
        if si_norm_mat is None:
            self.user_embed = nn.Parameter(torch.empty(self.n_users, args_config.dim))
        else:
            self.register_buffer('si_norm_mat', sparse_tensor(si_norm_mat), persistent=False)
        self.item_embed = nn.Parameter(torch.empty(self.n_items, args_config.dim))
        self.register_buffer('norm_adj', sparse_tensor(adj_mat), persistent=False)

    def build_user_embed(self):
        if hasattr(self, 'si_norm_mat'):
            return torch.sparse.mm(self.si_norm_mat, self.item_embed)
        return self.user_embed

    def split(self, embeddings):
        return embeddings[:self.n_users], embeddings[self.n_users:]

    def generate(self, split=True):
        embeddings = self.encode()
        return self.split(embeddings) if split else embeddings

    @staticmethod
    def rating(u_g_embeddings, i_g_embeddings):
        return u_g_embeddings @ i_g_embeddings.T

    def recommendation_loss(self, users, items, batch):
        negatives = batch['neg_items']
        if negatives.ndim == 2:
            negatives = negatives[:, 0]
        u, p, n = users[batch['users']], items[batch['pos_items']], items[negatives]
        difference = (u * (p - n)).sum(-1)
        # Exact QRec util.loss: -sum(log(sigmoid(difference) + 1e-7)).
        bpr = -torch.logaddexp(F.logsigmoid(difference),
                              torch.full_like(difference, math.log(1e-7))).sum()
        # QRec regularizes the propagated batch representations, with no / batch_size.
        reg = self.reg_weight * (u.square().sum() + p.square().sum() + n.square().sum()) / 2
        return bpr + reg

    def forward(self, batch=None, epoch=0):
        users, items = self.generate()
        rec = self.recommendation_loss(users, items, batch)
        cl = self.contrastive_loss(batch) * self.cl_rate if self.cl_rate else rec * 0.
        return rec + cl, rec, cl
