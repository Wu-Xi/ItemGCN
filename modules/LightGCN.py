import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
import numpy as np

class GraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, n_users, embedding0, interact_mat):
        super().__init__()
        if n_hops < 0:
            raise ValueError("n_hops must be positive")

        self.n_hops = n_hops
        self.n_users = n_users
        self.embedding0 = embedding0
        self.register_buffer("interact_mat", interact_mat, persistent=False)

    def forward(self, user_embed, item_embed):
        embeddings = torch.cat([user_embed, item_embed], dim=0)
        if self.embedding0 is True:
            layers = [embeddings]
        elif self.embedding0 is False:
            layers = []
        else:
            raise ValueError("n_hops must be bool")
        
        for _ in range(self.n_hops):
            embeddings = torch.sparse.mm(self.interact_mat, embeddings)
            layers.append(embeddings)
        embeddings = torch.stack(layers, dim=1)
        return embeddings[:self.n_users], embeddings[self.n_users:]

# python main.py --gnn lightgcn --lr 1e-03 --l2 1e-03 --dataset ali --pool mean --batch_size 2048 --gpu_id 0 --n_negs 1 --window_length 0
class LightGCN(nn.Module):
    requires_negative_sampling = True
    negative_sample_count = 1
    
    
    def __init__(self, data_config, args_config, adj_mat):
        super().__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.pool = args_config.pool
        self.ns = str(args_config.ns).lower()
        self.embedding0 = args_config.embedding0
        self.alpha = args_config.alpha
        self.negative_sample_count = 1 if self.ns == 'rns' else args_config.n_negs
        self.requires_history = self.ns == 'stabcf'
        if self.requires_history and (not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError('StabCF alpha must be finite and positive')
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        if self.ns == 'rns':
            self.negative_sampling = self.RNS_sampling
        elif self.ns =='stabcf':
            self.negative_sampling = self.StabCF_sampling
        elif self.ns == 'mixgcf':
            self.negative_sampling = self.MixGCF_sampling
        else:
            raise NotImplementedError(f"This {self.ns} is not found...")

        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         embedding0=self.embedding0,
                         interact_mat=self.sparse_norm_adj)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed)
        pos_embs = item_gcn_emb[pos_item]
        if self.ns == 'rns':
            negs_embs = self.RNS_sampling(item_gcn_emb, neg_item)
        elif self.ns == 'mixgcf':
            negs_embs = self.MixGCF_sampling(user_gcn_emb, item_gcn_emb, user, neg_item, pos_item)
        else:
            pos_embs, negs_embs = self.StabCF_sampling(
                user_gcn_emb, item_gcn_emb, user, neg_item, pos_item, batch['observed_pos_items'])
        batch_loss1,mf_loss1,emb_loss1=self.create_bpr_loss(user_gcn_emb[user], pos_embs, negs_embs)
        self._check_nan(batch_loss1)
        return batch_loss1,mf_loss1,emb_loss1

    def RNS_sampling(self, item_gcn_emb, neg_candidates):
        if neg_candidates.shape[1] == 1:
            neg_item = neg_candidates[:, 0]
        else:
            raise ValueError("RNS only need 1 neg")
        return item_gcn_emb[neg_item]

    def StabCF_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item, user_pos):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """Historical Positives Mixing"""
        his_e = torch.cat([item_gcn_emb[user_pos], p_e.unsqueeze(dim=1)], dim=1)  # [batch_size, n_pos+1, n_hops+1, channel]
        his_scores = (s_e.unsqueeze(dim=1) * his_e).sum(dim=-1)
        his_scores[:, -1, :] += math.log(self.alpha)
        p_e_ = (his_scores.softmax(dim=1).unsqueeze(dim=-1) * his_e).sum(dim=1)
        
        """User-Aware Negatives Mixing"""
        n_e = item_gcn_emb[neg_candidates]      # [batch_size, n_negs, n_hops+1, channel]
        indices_max_un = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1).argmax(dim=1)
        indices_max_pn = (p_e_.unsqueeze(dim=1) * n_e).sum(dim=-1).argmax(dim=1)
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        neg_items_embedding_hardest_un = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_un, :] 
        neg_items_embedding_hardest_pn = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_pn, :]
        
        neg_items_embedding = torch.stack([neg_items_embedding_hardest_un, neg_items_embedding_hardest_pn, p_e_], dim=1)
        nes_scores = (s_e.unsqueeze(dim=1) * neg_items_embedding).sum(dim=-1)
        nes_scores[:, -1, :] += math.log(self.alpha)
        n_e_ = (nes_scores.softmax(dim=1).unsqueeze(dim=-1) * neg_items_embedding).sum(dim=1)
        return p_e_, n_e_

    def MixGCF_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=-1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]
        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores)))

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:,0,:]) ** 2
                       + torch.norm(pos_gcn_embs[:,0,:]) ** 2
                       + torch.norm(neg_gcn_embs[:,0,:]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * (regularize0) / batch_size
        return mf_loss+emb_loss, mf_loss, emb_loss

class XLightGCN(nn.Module):
    requires_negative_sampling = True
    negative_sample_count = 1
    
    
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.si_norm_mat = si_norm_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.pool = args_config.pool
        self.ns = str(args_config.ns).lower()
        self.embedding0 = args_config.embedding0
        self.alpha = args_config.alpha
        self.negative_sample_count = 1 if self.ns == 'rns' else args_config.n_negs
        self.requires_history = self.ns == 'stabcf'
        if self.requires_history and (not math.isfinite(self.alpha) or self.alpha <= 0):
            raise ValueError('StabCF alpha must be finite and positive')
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        if self.ns == 'rns':
            self.negative_sampling = self.RNS_sampling
        elif self.ns =='stabcf':
            self.negative_sampling = self.StabCF_sampling
        elif self.ns == 'mixgcf':
            self.negative_sampling = self.MixGCF_sampling
        else:
            raise NotImplementedError(f"This {self.ns} is not found...")

        self._init_weight()
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.si_norm_mat = self._convert_sp_mat_to_sp_tensor(self.si_norm_mat).to(self.device)

    def build_user_embed(self):
        return torch.sparse.mm(self.si_norm_mat, self.item_embed)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         embedding0=self.embedding0,
                         interact_mat=self.sparse_norm_adj)

    def _check_nan(self, loss):
        if torch.isnan(loss):
            raise ValueError('Training loss is nan')

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']
        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.build_user_embed(), self.item_embed)
        pos_embs = item_gcn_emb[pos_item]
        if self.ns == 'rns':
            negs_embs = self.RNS_sampling(item_gcn_emb, neg_item)
        elif self.ns == 'mixgcf':
            negs_embs = self.MixGCF_sampling(user_gcn_emb, item_gcn_emb, user, neg_item, pos_item)
        else:
            pos_embs, negs_embs = self.StabCF_sampling(
                user_gcn_emb, item_gcn_emb, user, neg_item, pos_item, batch['observed_pos_items'])
        batch_loss1,mf_loss1,emb_loss1=self.create_bpr_loss(user_gcn_emb[user], pos_embs, negs_embs)
        self._check_nan(batch_loss1)
        return batch_loss1,mf_loss1,emb_loss1

    def RNS_sampling(self, item_gcn_emb, neg_candidates):
        if neg_candidates.shape[1] == 1:
            neg_item = neg_candidates[:, 0]
        else:
            raise ValueError("RNS only need 1 neg")
        return item_gcn_emb[neg_item]

    def StabCF_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item, user_pos):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """Historical Positives Mixing"""
        his_e = torch.cat([item_gcn_emb[user_pos], p_e.unsqueeze(dim=1)], dim=1)  # [batch_size, n_pos+1, n_hops+1, channel]
        his_scores = (s_e.unsqueeze(dim=1) * his_e).sum(dim=-1)
        his_scores[:, -1, :] += math.log(self.alpha)
        p_e_ = (his_scores.softmax(dim=1).unsqueeze(dim=-1) * his_e).sum(dim=1)
        
        """User-Aware Negatives Mixing"""
        n_e = item_gcn_emb[neg_candidates]      # [batch_size, n_negs, n_hops+1, channel]
        indices_max_un = (s_e.unsqueeze(dim=1) * n_e).sum(dim=-1).argmax(dim=1)
        indices_max_pn = (p_e_.unsqueeze(dim=1) * n_e).sum(dim=-1).argmax(dim=1)
        neg_items_emb_ = n_e.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        neg_items_embedding_hardest_un = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_un, :] 
        neg_items_embedding_hardest_pn = neg_items_emb_[[[i] for i in range(batch_size)],range(neg_items_emb_.shape[1]), indices_max_pn, :]
        
        neg_items_embedding = torch.stack([neg_items_embedding_hardest_un, neg_items_embedding_hardest_pn, p_e_], dim=1)
        nes_scores = (s_e.unsqueeze(dim=1) * neg_items_embedding).sum(dim=-1)
        nes_scores[:, -1, :] += math.log(self.alpha)
        n_e_ = (nes_scores.softmax(dim=1).unsqueeze(dim=-1) * neg_items_embedding).sum(dim=1)
        return p_e_, n_e_

    def MixGCF_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        batch_size = user.shape[0]
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        if self.pool != 'concat':
            s_e = self.pooling(s_e).unsqueeze(dim=1)

        """positive mixing"""
        seed = torch.rand(batch_size, 1, p_e.shape[1], 1).to(p_e.device)  # (0, 1)
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops, channel]
        n_e_ = seed * p_e.unsqueeze(dim=1) + (1 - seed) * n_e  # mixing

        """hop mixing"""
        scores = (s_e.unsqueeze(dim=1) * n_e_).sum(dim=-1)  # [batch_size, n_negs, n_hops+1]
        indices = torch.max(scores, dim=1)[1].detach()
        neg_items_emb_ = n_e_.permute([0, 2, 1, 3])  # [batch_size, n_hops+1, n_negs, channel]
        # [batch_size, n_hops+1, channel]
        return neg_items_emb_[[[i] for i in range(batch_size)],
                              range(neg_items_emb_.shape[1]), indices, :]

    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.build_user_embed(), self.item_embed)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]

        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=-1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]
        mf_loss = torch.mean(torch.log(1+torch.exp(neg_scores - pos_scores)))

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:,0,:]) ** 2
                       + torch.norm(pos_gcn_embs[:,0,:]) ** 2
                       + torch.norm(neg_gcn_embs[:,0,:]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * (regularize0) / batch_size
        return mf_loss+emb_loss, mf_loss, emb_loss

class AHNS(nn.Module):
    requires_negative_sampling = True
    
    def __init__(self, data_config, args_config, adj_mat):
        super().__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.embedding0 = args_config.embedding0

        self.simi = args_config.simi
        self.p = args_config.p
        self.alpha = args_config.alpha
        self.beta = args_config.beta

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self._init_weight()
        self.user_embed = nn.Parameter(self.user_embed)
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.user_embed = initializer(torch.empty(self.n_users, self.emb_size))
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        # [n_users+n_items, n_users+n_items]
        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         embedding0=self.embedding0,
                         interact_mat=self.sparse_norm_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def forward(self, batch=None, epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed)


        neg_gcn_embs = []
        for k in range(1):
            neg_gcn_embs.append(self.adaptive_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item))
        neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def adaptive_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        
        s_e = s_e.mean(dim=1)  # [batch_size, channel]
        p_e = p_e.mean(dim=1)  # [batch_size, channel]
        n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]
                
        p_scores = self.similarity(s_e, p_e).unsqueeze(dim=1) # [batch_size, 1]
        n_scores = self.similarity(s_e.unsqueeze(dim=1), n_e) # [batch_size, n_negs]

        scores = torch.abs(n_scores - self.beta * (p_scores + self.alpha).pow(self.p + 1))

        """adaptive negative sampling"""
        indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze(-1)
        
        return item_gcn_emb[neg_item]
        
    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def similarity(self, user_embeddings, item_embeddings):
        # [-1, n_hops, channel]
        if self.simi == 'ip':
            return (user_embeddings * item_embeddings).sum(dim=-1)
        elif self.simi == 'cos':
            return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
        elif self.simi == 'ed':
            return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
        else:  # ip
            return (user_embeddings * item_embeddings).sum(dim=-1)

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.user_embed, self.item_embed)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]
        
        batch_size = user.shape[0]
        
        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, 1, -1)
        
        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(dim=1) * neg_e).sum(dim=-1)
        
        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))
        
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss

class XAHNS(nn.Module):
    requires_negative_sampling = True
    
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.si_norm_mat = si_norm_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.pool = args_config.pool
        self.n_negs = args_config.n_negs
        self.embedding0 = args_config.embedding0

        self.simi = args_config.simi
        self.p = args_config.p
        self.alpha = args_config.alpha
        self.beta = args_config.beta

        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self._init_weight()
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.si_norm_mat = self._convert_sp_mat_to_sp_tensor(self.si_norm_mat).to(self.device)

    def _init_model(self):
        return GraphConv(n_hops=self.context_hops,
                         n_users=self.n_users,
                         embedding0=self.embedding0,
                         interact_mat=self.sparse_norm_adj)

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo()
        i = torch.LongTensor([coo.row, coo.col])
        v = torch.from_numpy(coo.data).float()
        return torch.sparse.FloatTensor(i, v, coo.shape)

    def build_user_embed(self):
        return torch.sparse.mm(self.si_norm_mat, self.item_embed)

    def forward(self, batch=None, epoch=0):
        user = batch['users']
        pos_item = batch['pos_items']
        neg_item = batch['neg_items']  # [batch_size, n_negs * K]

        # user_gcn_emb: [n_users, channel]
        # item_gcn_emb: [n_users, channel]
        user_gcn_emb, item_gcn_emb = self.gcn(self.build_user_embed(), self.item_embed)


        neg_gcn_embs = []
        for k in range(1):
            neg_gcn_embs.append(self.adaptive_negative_sampling(user_gcn_emb, item_gcn_emb,
                                                                user,
                                                                neg_item[:, k * self.n_negs: (k + 1) * self.n_negs],
                                                                pos_item))
        neg_gcn_embs = torch.stack(neg_gcn_embs, dim=1)

        return self.create_bpr_loss(user, user_gcn_emb[user], item_gcn_emb[pos_item], neg_gcn_embs)

    def adaptive_negative_sampling(self, user_gcn_emb, item_gcn_emb, user, neg_candidates, pos_item):
        s_e, p_e = user_gcn_emb[user], item_gcn_emb[pos_item]  # [batch_size, n_hops+1, channel]
        n_e = item_gcn_emb[neg_candidates]  # [batch_size, n_negs, n_hops+1, channel]
        
        s_e = s_e.mean(dim=1)  # [batch_size, channel]
        p_e = p_e.mean(dim=1)  # [batch_size, channel]
        n_e = n_e.mean(dim=2)  # [batch_size, n_negs, channel]
                
        p_scores = self.similarity(s_e, p_e).unsqueeze(dim=1) # [batch_size, 1]
        n_scores = self.similarity(s_e.unsqueeze(dim=1), n_e) # [batch_size, n_negs]
        scores = torch.abs(n_scores - self.beta * (p_scores + self.alpha).pow(self.p + 1))

        """adaptive negative sampling"""
        indices = torch.min(scores, dim=1)[1].detach()  # [batch_size]
        neg_item = torch.gather(neg_candidates, dim=1, index=indices.unsqueeze(-1)).squeeze(-1)
        
        return item_gcn_emb[neg_item]
        
    def pooling(self, embeddings):
        # [-1, n_hops, channel]
        if self.pool == 'mean':
            return embeddings.mean(dim=1)
        elif self.pool == 'sum':
            return embeddings.sum(dim=1)
        elif self.pool == 'concat':
            return embeddings.view(embeddings.shape[0], -1)
        else:  # final
            return embeddings[:, -1, :]

    def similarity(self, user_embeddings, item_embeddings):
        # [-1, n_hops, channel]
        if self.simi == 'ip':
            return (user_embeddings * item_embeddings).sum(dim=-1)
        elif self.simi == 'cos':
            return F.cosine_similarity(user_embeddings, item_embeddings, dim=-1)
        elif self.simi == 'ed':
            return ((user_embeddings - item_embeddings) ** 2).sum(dim=-1)
        else:  # ip
            return (user_embeddings * item_embeddings).sum(dim=-1)

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.build_user_embed(), self.item_embed)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split:
            return user_gcn_emb, item_gcn_emb
        else:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        # user_gcn_emb: [batch_size, n_hops+1, channel]
        # pos_gcn_embs: [batch_size, n_hops+1, channel]
        # neg_gcn_embs: [batch_size, K, n_hops+1, channel]
        
        batch_size = user.shape[0]
        
        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs.view(-1, neg_gcn_embs.shape[2], neg_gcn_embs.shape[3])).view(batch_size, 1, -1)
        
        pos_scores = (u_e * pos_e).sum(dim=-1)
        neg_scores = (u_e.unsqueeze(dim=1) * neg_e).sum(dim=-1)
        
        mf_loss = torch.mean(torch.log(1 + torch.exp(neg_scores - pos_scores.unsqueeze(dim=1)).sum(dim=1)))
        
        # cul regularizer
        regularize = (torch.norm(user_gcn_emb[:, 0, :]) ** 2
                      + torch.norm(pos_gcn_embs[:, 0, :]) ** 2
                      + torch.norm(neg_gcn_embs[:, :, 0, :]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * regularize / batch_size

        return mf_loss + emb_loss, mf_loss, emb_loss
    

class DirectAU(nn.Module):
    requires_negative_sampling = False

    def __init__(self, data_config, args_config, adj_mat, embedding0=None):
        super().__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_size = args_config.dim
        self.embedding0 = args_config.embedding0 if embedding0 is None else embedding0
        
        self.encoder = getattr(args_config, 'encoder', 'mf').lower()
        self.context_hops = args_config.context_hops
        gamma = getattr(args_config, 'gamma', None)
        self.gamma = 1.0 if gamma is None else gamma
        if self.encoder not in ('mf', 'lightgcn'):
            raise ValueError('DirectAU encoder must be mf or lightgcn')
        if self.context_hops < 0:
            raise ValueError('context_hops must be nonnegative')

        self.user_embed = nn.Parameter(torch.empty(self.n_users, self.emb_size))
        self.item_embed = nn.Parameter(torch.empty(self.n_items, self.emb_size))
        nn.init.xavier_normal_(self.user_embed)
        nn.init.xavier_normal_(self.item_embed)
        coo = adj_mat.tocoo()
        indices = torch.from_numpy(np.stack([coo.row, coo.col])).long()
        adjacency = torch.sparse_coo_tensor(
            indices, torch.as_tensor(coo.data, dtype=torch.float32), coo.shape).coalesce()
        # The graph is dataset state, not a learned checkpoint parameter.
        self.register_buffer('sparse_norm_adj', adjacency, persistent=False)
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")

    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        # There are no within-side pairs in a singleton tail batch.
        if x.shape[0] < 2:
            return x.sum() * 0.0
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def generate(self, split=True):
        embeddings = torch.cat([self.user_embed, self.item_embed], dim=0)
        if self.encoder == 'lightgcn':
            if self.embedding0 == True:
                layers = [embeddings]
            elif self.embedding0 ==False:
                layers=[]
            else:
                raise ValueError(f"This {self.embedding0} value not true")
            for _ in range(self.context_hops):
                embeddings = torch.sparse.mm(self.sparse_norm_adj, embeddings)
                layers.append(embeddings)
            embeddings = torch.stack(layers, dim=0).mean(dim=0)
        # Original full-sort evaluation uses unnormalized dot products.
        if split:
            return embeddings[:self.n_users], embeddings[self.n_users:]
        return embeddings

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def forward(self, batch=None, epoch=0):
        users, items = self.generate()
        user_e, item_e = users[batch['users']], items[batch['pos_items']]
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align + uniform, align, uniform


class XDirectAU(nn.Module):
    requires_negative_sampling = False

    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__()
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.emb_size = args_config.dim
        self.encoder = getattr(args_config, 'encoder', 'lightgcn').lower()
        self.context_hops = args_config.context_hops
        gamma = getattr(args_config, 'gamma', None)
        self.gamma = 1.0 if gamma is None else gamma
        self.embedding0 = args_config.embedding0
        self.beta = getattr(args_config, 'beta', None)
        
        if self.encoder not in ('mf', 'lightgcn'):
            raise ValueError('DirectAU_ encoder must be mf or lightgcn')
        if self.context_hops < 0:
            raise ValueError('context_hops must be nonnegative')

        self.item_embed = nn.Parameter(torch.empty(self.n_items, self.emb_size))
        nn.init.xavier_normal_(self.item_embed)
        
        coo = adj_mat.tocoo()
        indices = torch.from_numpy(np.stack([coo.row, coo.col])).long()
        adjacency = torch.sparse_coo_tensor(
            indices, torch.as_tensor(coo.data, dtype=torch.float32), coo.shape).coalesce()
        self.register_buffer('sparse_norm_adj', adjacency, persistent=False)
        
        coo = si_norm_mat.tocoo()
        indices = torch.from_numpy(np.stack([coo.row, coo.col])).long()
        adjacency = torch.sparse_coo_tensor(
            indices, torch.as_tensor(coo.data, dtype=torch.float32), coo.shape).coalesce()
        self.register_buffer('si_norm_mat', adjacency, persistent=False)
        self.register_buffer('si_norm_mat_t', self.si_norm_mat.transpose(0, 1).coalesce(), persistent=False)
        
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")

    def build_user_embed(self):
        return torch.sparse.mm(self.si_norm_mat, self.item_embed)
    
    @staticmethod
    def alignment(x, y):
        x, y = F.normalize(x, dim=-1), F.normalize(y, dim=-1)
        return (x - y).norm(p=2, dim=1).pow(2).mean()

    @staticmethod
    def uniformity(x):
        # There are no within-side pairs in a singleton tail batch.
        if x.shape[0] < 2:
            return x.sum() * 0.0
        x = F.normalize(x, dim=-1)
        return torch.pdist(x, p=2).pow(2).mul(-2).exp().mean().log()

    def generate(self, split=True):
        embeddings = torch.cat([self.build_user_embed(), self.item_embed], dim=0)
        if self.encoder == 'lightgcn':
            if self.embedding0 == True:
                layers = [embeddings]
            elif self.embedding0 ==False:
                layers=[]
            else:
                raise ValueError(f"This {self.embedding0} value not true")
            for _ in range(self.context_hops):
                embeddings = torch.sparse.mm(self.sparse_norm_adj, embeddings)
                layers.append(embeddings)
            embeddings = torch.stack(layers, dim=0).mean(dim=0)
        # Original full-sort evaluation uses unnormalized dot products.
        if split:
            return embeddings[:self.n_users], embeddings[self.n_users:]
        return embeddings

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def forward(self, batch=None, epoch=0):
        users, items = self.generate()
        user_e, item_e = users[batch['users']], items[batch['pos_items']]
        align = self.alignment(user_e, item_e)
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align + uniform, align, uniform


class GraphAU(DirectAU):
    def __init__(self, data_config, args_config, adj_mat):
        super().__init__(data_config, args_config, adj_mat)
        self.encoder = 'mf'
        gamma = getattr(args_config, 'gamma', None)
        self.gamma = 0.4 if gamma is None else gamma
        self.layers = getattr(args_config, 'graphau_layers', 4)
        self.decay_base = getattr(args_config, 'decaying_base', 1.4)
        if self.layers < 1:
            raise ValueError('graphau_layers must be at least 1')
        if self.decay_base <= 0:
            raise ValueError('decaying_base must be positive')
        # Match the original GraphAU standard-normal initialization.
        nn.init.normal_(self.user_embed)
        nn.init.normal_(self.item_embed)
        self.register_buffer('decay_weight', torch.tensor(
            [self.decay_base ** layer for layer in range(self.layers)]))

    def build_user_embed(self):
        return self.user_embed

    def forward(self, batch=None, epoch=0):
        users, items = batch['users'], batch['pos_items']
        user_embeddings = self.build_user_embed()
        user_e, item_e = user_embeddings[users], self.item_embed[items]
        alignments = [self.alignment(user_e, item_e)]
        embeddings = torch.cat([user_embeddings, self.item_embed], dim=0)
        # Original layers includes the zero-hop term: layers=2 means one hop.
        # Propagate incrementally instead of recomputing every shorter path.
        for _ in range(1, self.layers):
            embeddings = torch.sparse.mm(self.sparse_norm_adj, embeddings)
            user_agg = embeddings[users]
            item_agg = embeddings[self.n_users + items]
            alignments.append((self.alignment(user_e, item_agg)
                               + self.alignment(user_agg, item_e)) / 2)
        # Divide by layer count, NOT by the sum of weights (source behavior).
        align = (self.decay_weight * torch.stack(alignments)).mean()
        uniform = self.gamma * (self.uniformity(user_e) + self.uniformity(item_e)) / 2
        return align + uniform, align, uniform


class XGraphAU(XDirectAU):
    """GraphAU with U = BQ and no trainable user embedding table."""
    def __init__(self, data_config, args_config, adj_mat, si_norm_mat):
        super().__init__(data_config, args_config, adj_mat, si_norm_mat)
        self.encoder = 'mf'
        self.gamma = 0.4 if args_config.gamma is None else args_config.gamma
        self.layers = args_config.graphau_layers
        self.decay_base = args_config.decaying_base
        if self.layers < 1 or self.decay_base <= 0:
            raise ValueError('graphau_layers and decaying_base must be positive')
        nn.init.normal_(self.item_embed)
        self.register_buffer('decay_weight', torch.tensor(
            [self.decay_base ** layer for layer in range(self.layers)]))

    forward = GraphAU.forward


# Keep the existing model import location available.
from .SimGCL import SimGCL, XSimGCL
from .SGL import SGL, XSGL
from .RecDCL import RecDCL, XRecDCL
