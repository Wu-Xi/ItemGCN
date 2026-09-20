"""RecDCL adapted from the local RecBole implementation to MixGCF."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class TailBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        # RecBole's original BN cannot train on a singleton. Keep the tail,
        # using existing running statistics without updating them in this case.
        if self.training and x.shape[0] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var,
                                self.weight, self.bias, False, 0., self.eps)
        return super().forward(x)


class RecDCL(nn.Module):
    requires_negative_sampling = False

    def __init__(self, data_config, args_config, adj_mat, si_norm_mat=None):
        super().__init__()
        self.n_users, self.n_items = data_config['n_users'], data_config['n_items']
        self.embedding_size = args_config.dim
        self.encoder = args_config.encoder.lower()
        self.context_hops = args_config.context_hops
        for name, default in dict(a=1., polyc=1e-7, degree=4, poly_coeff=.2,
                                  bt_coeff=.01, all_bt_coeff=1., mom_coeff=10., momentum=.3).items():
            setattr(self, name, getattr(args_config, name, default))
        if si_norm_mat is None:
            self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        else:
            from .graph_contrastive import sparse_tensor
            self.register_buffer('si_norm_mat', sparse_tensor(si_norm_mat), persistent=False)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        coo = adj_mat.tocoo()
        adjacency = torch.sparse_coo_tensor(
            torch.from_numpy(np.stack([coo.row, coo.col])).long(),
            torch.as_tensor(coo.data, dtype=torch.float32), coo.shape).coalesce()
        self.register_buffer('sparse_norm_adj', adjacency, persistent=False)
        self.bn = TailBatchNorm(self.embedding_size, affine=False)
        d = self.embedding_size
        self.projector = nn.Sequential(
            nn.Linear(d, d, bias=False), TailBatchNorm(d), nn.ReLU(inplace=True),
            nn.Linear(d, d, bias=False), TailBatchNorm(d), nn.ReLU(inplace=True),
            nn.Linear(d, d, bias=False))
        for layer in self.modules():
            if isinstance(layer, (nn.Embedding, nn.Linear)):
                nn.init.xavier_normal_(layer.weight)
                if isinstance(layer, nn.Linear) and layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Source creates predictor after Xavier initialization: keep its defaults.
        self.predictor = nn.Linear(d, d)
        self.register_buffer('u_target_his', torch.randn(self.n_users, d))
        self.register_buffer('i_target_his', torch.randn(self.n_items, d))

    def build_user_embed(self):
        if hasattr(self, 'si_norm_mat'):
            return torch.sparse.mm(self.si_norm_mat, self.item_embedding.weight)
        return self.user_embedding.weight

    def generate(self, split=True):
        embeddings = torch.cat([self.build_user_embed(), self.item_embedding.weight])
        if self.encoder == 'lightgcn':
            layers = [embeddings]
            for _ in range(self.context_hops):
                embeddings = torch.sparse.mm(self.sparse_norm_adj, embeddings)
                layers.append(embeddings)
            embeddings = torch.stack(layers).mean(0)
        if split:
            return embeddings[:self.n_users], embeddings[self.n_users:]
        return embeddings

    @staticmethod
    def rating(u_g_embeddings, i_g_embeddings):
        return u_g_embeddings @ i_g_embeddings.T

    @staticmethod
    def off_diagonal(x):
        return x[~torch.eye(x.shape[0], device=x.device, dtype=torch.bool)]

    def bt(self, x, y):
        x, y = self.projector(x), self.projector(y)
        c = self.bn(x).T @ self.bn(y) / x.shape[0]
        return ((c.diagonal() - 1).square().sum()
                + self.bt_coeff * self.off_diagonal(c).square().sum()) / self.embedding_size

    def poly_feature(self, x):
        x = self.projector(x)
        # Deliberately no batch-size divisor: this is the source formula.
        xx = self.bn(x).T @ self.bn(x)
        return ((self.a * xx + self.polyc) ** self.degree).mean().log()

    @staticmethod
    def loss_fn(p, z):
        return -F.cosine_similarity(p, z.detach(), dim=-1).mean()

    def forward(self, batch=None, epoch=0):
        users, items = self.generate()
        uid, iid = batch['users'], batch['pos_items']
        u, i = users[uid], items[iid]
        with torch.no_grad():
            u_target = self.momentum * self.u_target_his[uid] + (1 - self.momentum) * u.detach()
            i_target = self.momentum * self.i_target_his[iid] + (1 - self.momentum) * i.detach()
            if self.training:
                # Repeated IDs have identical encoder outputs; write each once.
                unique_u, unique_i = uid.unique(), iid.unique()
                self.u_target_his[unique_u] = users[unique_u].detach()
                self.i_target_his[unique_i] = items[unique_i].detach()
        un, inn = F.normalize(u, dim=-1), F.normalize(i, dim=-1)
        zero = (u.sum() + i.sum()) * 0.
        bt = self.all_bt_coeff * self.bt(un, inn) if self.all_bt_coeff else zero
        poly = self.poly_coeff * (self.poly_feature(un) + self.poly_feature(inn)) / 2 if self.poly_coeff else zero
        mom = self.mom_coeff * (self.loss_fn(self.predictor(u), i_target)
                               + self.loss_fn(self.predictor(i), u_target)) / 2 if self.mom_coeff else zero
        return bt + poly + mom, bt + poly, mom


class XRecDCL(RecDCL):
    """Item-only variant; user history targets remain non-trainable buffers."""
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__(data_config, args_config, adj_mat, si_norm_mat)
