import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from Utils.Constraint import *
from model.gin import Encoder


def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


class DCGLC(nn.Module):
    def __init__(self, args, dataset_num_features, n_label, device, alpha=1.0, gamma=.1):
        super(DCGLC, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.device = device
        self.prior = args.prior
        self.d = args.d
        cluster_emb = self.d * n_label
        self.s = None
        self.eta = args.eta
        self.n_clusters=n_label

        self.encoder = encoder(args, dataset_num_features, n_label)
        self.cluster_layer = Parameter(torch.Tensor(n_label, cluster_emb))

        # Subspace bases proxy
        self.D = Parameter(torch.Tensor(cluster_emb, cluster_emb))


    def forward(self, x, edge_index, batch, num_graphs, n_aug=0):

        s = None
        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(self.device)

        z, y = self.encoder(x, edge_index, batch, num_graphs)

        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        # Calculate subspace affinity
        for i in range(self.n_clusters):

            si = torch.sum(torch.pow(torch.mm(z, self.D[:, i * self.d:(i + 1) * self.d]), 2), 1, keepdim=True)
            if s is None:
                s = si
            else:
                s = torch.cat((s, si), 1)
        s = (s + self.eta * self.d) / ((self.eta + 1) * self.d)
        s = (s.t() / torch.sum(s, 1)).t()
        return z, q, s, y


    def get_results(self, loader):
        embedding = []
        cluster = []
        y = []
        cluster_subspace= []
        cluster_merge=[]
        with torch.no_grad():
            for data in loader:
                data, data_aug = data
                data = data.to(self.device)
                x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(self.device)
                z, q, s, _ = self.forward(x, edge_index, batch, num_graphs)
                embedding.append(z.cpu().numpy())
                cluster.append(q.cpu().numpy())
                cluster_subspace.append(s.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        cluster = np.concatenate(cluster, 0)
        cluster_subspace = np.concatenate(cluster_subspace, 0)
        y = np.concatenate(y, 0)
        return embedding, cluster, cluster_subspace, y

    def loss_cal(self, x, x_aug):
        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        # Constraints
        d_cons1 = D_constraint1(self.device)
        d_cons2 = D_constraint2(self.device)
        loss_d1 = d_cons1(self.D)
        loss_d2 = d_cons2(self.D, self.d, self.n_clusters)
        loss = loss + loss_d1 + loss_d2

        return loss

EPS = 1e-15
def loss_balance_entropy(prob, *kwargs):
    prob = prob.clamp(EPS)
    entropy = prob * prob.log()

    # return negative entropy to maximize it
    if entropy.ndim == 1:
        return entropy.sum()
    elif entropy.ndim == 2:
        return entropy.sum(dim=1).mean()
    else:
        raise ValueError(f'Probability is {entropy.ndim}-d')


class encoder(nn.Module):
    def __init__(self, args, dataset_num_features, device, alpha=1.0):
        super(encoder, self).__init__()

        self.alpha = alpha
        self.prior = args.prior
        self.device = device

        self.embedding_dim = mi_units = args.hidden_dim * args.num_gc_layers
        self.cluster_embedding = Cluster(self.embedding_dim, args.cluster_emb)
        self.encoder = Encoder(dataset_num_features, args.hidden_dim, args.num_gc_layers, self.device)

        self.proj_head = nn.ModuleList([nn.Sequential(nn.Linear(self.embedding_dim, self.embedding_dim),
                                                      nn.LeakyReLU(inplace=True),
                                                      nn.Linear(self.embedding_dim, self.embedding_dim)) for _ in
                                        range(5)])
        self.local_d = FF(self.embedding_dim)
        self.global_d = FF(self.embedding_dim)

        self.init_emb()

    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, x, edge_index, batch, num_graphs, n_aug=0):

        if x is None or len(x.shape) == 1 or x.shape[1] == 0:
            x = torch.ones(batch.shape[0]).to(device)

        y, M = self.encoder(x, edge_index, batch)

        y = self.proj_head[n_aug](y)

        g_enc = self.global_d(y)
        l_enc = self.local_d(M)

        # dimension reduction layer
        z = self.cluster_embedding(g_enc)


        return z, g_enc

    def get_results(self, loader):
        embedding = []
        y = []

        with torch.no_grad():
            for data in loader:
                data, data_aug = data
                data = data.to(device)
                # data_aug = data_aug.to(device)
                x, edge_index, batch, num_graphs = data.x, data.edge_index, data.batch, data.num_graphs
                if x is None:
                    x = torch.ones((batch.shape[0], 1)).to(device)
                z, _ = self.forward(x, edge_index, batch, num_graphs)
                embedding.append(z.cpu().numpy())
                y.append(data.y.cpu().numpy())
        embedding = np.concatenate(embedding, 0)
        y = np.concatenate(y, 0)
        return embedding, y

    def loss_cal(self, x, x_aug):

        T = 0.2
        batch_size, _ = x.size()
        x_abs = x.norm(dim=1)
        x_aug_abs = x_aug.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss).mean()

        return loss



class Cluster(nn.Module):
    def __init__(self, input_dim, cluster_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, int(input_dim/2)),
            nn.LeakyReLU(),
            nn.Linear(int(input_dim/2), cluster_dim),
            nn.LeakyReLU(),
        )
        self.linear_shortcut = nn.Linear(input_dim, cluster_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)

class FF(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU(),
            nn.Linear(input_dim, input_dim),

            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)


    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)