"""SimGCL translated from QRec/model/ranking/SimGCL.py."""
import torch
from torch import nn
from torch.nn import functional as F
from .graph_contrastive import GraphContrastive, info_nce


class SimGCL(GraphContrastive):
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat=None):
        super().__init__(data_config, args_config, adj_mat, si_norm_mat)
        self.eps = args_config.eps
        if hasattr(self, 'user_embed'):
            nn.init.xavier_uniform_(self.user_embed)
        nn.init.xavier_uniform_(self.item_embed)

    def encode(self, perturbed=False):
        embeddings = torch.cat([self.build_user_embed(), self.item_embed])
        layers = []  # QRec SimGCL excludes the initial embeddings.
        for _ in range(self.n_layers):
            embeddings = torch.sparse.mm(self.norm_adj, embeddings)
            if perturbed:
                noise = F.normalize(torch.rand_like(embeddings), dim=-1, eps=1e-6)
                embeddings = embeddings + embeddings.sign() * noise * self.eps
            layers.append(embeddings)
        return torch.stack(layers).mean(0)

    def contrastive_loss(self, batch):
        u1, i1 = self.split(self.encode(perturbed=True))
        u2, i2 = self.split(self.encode(perturbed=True))
        users, items = batch['users'].unique(), batch['pos_items'].unique()
        return (info_nce(u1[users], u2[users], self.temperature)
                + info_nce(i1[items], i2[items], self.temperature))


class XSimGCL(SimGCL):
    """Item-only variant: initial user representations are B @ Q."""
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__(data_config, args_config, adj_mat, si_norm_mat)
