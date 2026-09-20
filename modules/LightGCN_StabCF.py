import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# python main.py --gnn igcn --dataset ali --pool mean --gpu_id 0 --n_negs 1 --ns rns
class ItemGraphConv(nn.Module):
    """
    Graph Convolutional Network
    """
    def __init__(self, n_hops, embedding0, interact_mat, si_norm_mat):
        super().__init__()
        if n_hops < 0:
            raise ValueError("n_hops must be positive")

        self.n_hops = n_hops
        self.embedding0 = embedding0
        self.register_buffer("interact_mat", interact_mat, persistent=False)
        self.register_buffer("si_norm_mat", si_norm_mat, persistent=False)
        
    def build_user_embed(self, embs):
        return torch.sparse.mm(self.si_norm_mat, embs)
    
    def forward(self, embeddings):
        if self.embedding0 is True:
            user_embs, item_embs = [self.build_user_embed(embeddings)], [embeddings]
        elif self.embedding0 is False:
            user_embs, item_embs  = [], []
        else:
            raise ValueError("split must be bool")
        
        for _ in range(self.n_hops):
            embeddings = torch.sparse.mm(self.interact_mat, embeddings)
            agg_user = self.build_user_embed(embeddings)
            item_embs.append(embeddings)
            user_embs.append(agg_user)
            
        return torch.stack(user_embs, dim=1), torch.stack(item_embs, dim=1)

class StabCF2(nn.Module):
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
        self.embedding0 = args_config.embedding0
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

        self._init_weight()
        self.item_embed = nn.Parameter(self.item_embed)
        self.gcn = self._init_model()

    def _init_weight(self):
        initializer = nn.init.xavier_uniform_
        self.item_embed = initializer(torch.empty(self.n_items, self.emb_size))

        self.sparse_norm_adj = self._convert_sp_mat_to_sp_tensor(self.adj_mat).to(self.device)
        self.si_norm_mat = self._convert_sp_mat_to_sp_tensor(self.si_norm_mat).to(self.device)

    def _init_model(self):
        return ItemGraphConv(n_hops=self.context_hops,
                             embedding0=self.embedding0,
                             interact_mat=self.sparse_norm_adj,
                             si_norm_mat=self.si_norm_mat)

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
        
        user_gcn_emb, item_gcn_emb = self.gcn(self.item_embed)
        
        negs_embs = self.RNS_sampling(item_gcn_emb,neg_item)
        batch_loss1,mf_loss1,emb_loss1=self.create_bpr_loss(user_gcn_emb[user], item_gcn_emb[pos_item], negs_embs)
        self._check_nan(batch_loss1)
        return batch_loss1,mf_loss1,emb_loss1
    
    def RNS_sampling(self, item_gcn_emb, neg_candidates):
        if neg_candidates.shape[1] == 1:
            neg_item = neg_candidates[:, 0]
        else:
            raise ValueError("RNS only need 1 neg")
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

    def generate(self, split=True):
        user_gcn_emb, item_gcn_emb = self.gcn(self.item_embed)
        user_gcn_emb, item_gcn_emb = self.pooling(user_gcn_emb), self.pooling(item_gcn_emb)
        if split is True:
            return user_gcn_emb, item_gcn_emb
        elif split is False:
            return torch.cat([user_gcn_emb, item_gcn_emb], dim=0)
        else:
            raise ValueError("split must be bool")

    def rating(self, u_g_embeddings=None, i_g_embeddings=None):
        return torch.matmul(u_g_embeddings, i_g_embeddings.t())

    def create_bpr_loss(self, user_gcn_emb, pos_gcn_embs, neg_gcn_embs):
        batch_size = user_gcn_emb.shape[0]

        u_e = self.pooling(user_gcn_emb)
        pos_e = self.pooling(pos_gcn_embs)
        neg_e = self.pooling(neg_gcn_embs)

        pos_scores = torch.sum(torch.mul(u_e, pos_e), axis=-1)
        neg_scores = torch.sum(torch.mul(u_e, neg_e), axis=-1)  # [batch_size, K]
        mf_loss = F.softplus(neg_scores - pos_scores).mean()

        # cul regularizer
        regularize0 = (torch.norm(user_gcn_emb[:,0,:]) ** 2
                       + torch.norm(pos_gcn_embs[:,0,:]) ** 2
                       + torch.norm(neg_gcn_embs[:,0,:]) ** 2) / 2  # take hop=0
        emb_loss = self.decay * (regularize0) / batch_size
        return mf_loss+emb_loss, mf_loss, emb_loss


