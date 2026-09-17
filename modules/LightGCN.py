import torch
import torch.nn as nn
import os

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
    def __init__(self, data_config, args_config, adj_mat):
        super(LightGCN, self).__init__()

        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.adj_mat = adj_mat
        self.decay = args_config.l2
        self.emb_size = args_config.dim
        self.context_hops = args_config.context_hops
        self.pool = args_config.pool
        self.embedding0 = args_config.embedding0
        print(f"This is {os.path.basename(__file__)}.{self.__class__.__name__} working...")
        self.device = torch.device("cuda:0") if args_config.cuda else torch.device("cpu")

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
        negs_embs = self.RNS_sampling(item_gcn_emb, neg_item)
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
