"""SGL translated from QRec, including its joint user/item contrastive loss."""
import random
import numpy as np
import scipy.sparse as sp
import torch
from torch import nn
from .graph_contrastive import GraphContrastive, info_nce, sparse_tensor


class SGL(GraphContrastive):
    requires_interaction_matrix = True

    def __init__(self, data_config, args_config, adj_mat, interaction_matrix, si_norm_mat=None):
        super().__init__(data_config, args_config, adj_mat, si_norm_mat)
        self.drop_rate, self.aug_type = args_config.drop_rate, args_config.aug_type
        interactions = interaction_matrix.tocoo()
        self.edge_users = interactions.row.copy()
        self.edge_items = interactions.col.copy()
        # MixGCF supplies unique binary training interactions.
        self.view_count = self.n_layers if self.aug_type == 2 else 1
        for side in (1, 2):
            for layer in range(self.view_count):
                self.register_buffer('view_{}_{}'.format(side, layer), None, persistent=False)
        # QRec SGL inherits DeepRecommender's truncated-normal initialization.
        if hasattr(self, 'user_embed'):
            nn.init.trunc_normal_(self.user_embed, std=.005, a=-.01, b=.01)
        nn.init.trunc_normal_(self.item_embed, std=.005, a=-.01, b=.01)

    def augmented_adjacency(self):
        if self.drop_rate == 0:
            return self.norm_adj
        users, items = self.edge_users, self.edge_items
        if self.aug_type == 0:
            keep_u, keep_i = np.ones(self.n_users, bool), np.ones(self.n_items, bool)
            keep_u[random.sample(range(self.n_users), int(self.n_users * self.drop_rate))] = False
            keep_i[random.sample(range(self.n_items), int(self.n_items * self.drop_rate))] = False
            keep = keep_u[users] & keep_i[items]
        else:
            keep = random.sample(range(len(users)), int(len(users) * (1 - self.drop_rate)))
        users, items = users[keep], items[keep] + self.n_users
        row, col = np.concatenate([users, items]), np.concatenate([items, users])
        graph = sp.csr_matrix((np.ones(len(row), np.float32), (row, col)), shape=self.norm_adj.shape)
        degree = np.asarray(graph.sum(1)).ravel()
        inverse = np.zeros_like(degree)
        np.power(degree, -.5, out=inverse, where=degree > 0)
        diagonal = sp.diags(inverse)
        graph = diagonal @ graph @ diagonal
        return sparse_tensor(graph).to(device=self.item_embed.device, dtype=self.item_embed.dtype)

    def on_train_epoch_start(self):
        # Only graphs are reused within an epoch, never embedding computation graphs.
        if self.cl_rate:
            for layer in range(self.view_count):
                for side in (1, 2):
                    setattr(self, 'view_{}_{}'.format(side, layer), self.augmented_adjacency())

    def encode(self, side=None):
        embeddings = torch.cat([self.build_user_embed(), self.item_embed])
        layers = [embeddings]
        for layer in range(self.n_layers):
            graph = self.norm_adj if side is None else getattr(
                self, 'view_{}_{}'.format(side, layer if self.aug_type == 2 else 0))
            embeddings = torch.sparse.mm(graph, embeddings)
            layers.append(embeddings)
        return torch.stack(layers).mean(0)

    def contrastive_loss(self, batch):
        if self.view_1_0 is None:
            self.on_train_epoch_start()
        u1, i1 = self.split(self.encode(side=1))
        u2, i2 = self.split(self.encode(side=2))
        users, items = batch['users'].unique(), batch['pos_items'].unique()
        # trainModel calls calc_ssl_loss_v3, not the separate-group alternative.
        first = torch.cat([u1[users], i1[items]])
        second = torch.cat([u2[users], i2[items]])
        return info_nce(first, second, self.temperature)


class XSGL(SGL):
    """Item-only variant: initial user representations are B @ Q."""
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat, interaction_matrix):
        super().__init__(data_config, args_config, adj_mat, interaction_matrix, si_norm_mat)
