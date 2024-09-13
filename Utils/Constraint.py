import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class D_constraint1(torch.nn.Module):

    def __init__(self, device):
        super(D_constraint1, self).__init__()
        self.device = device

    def forward(self, d):
        I = torch.eye(d.shape[0]).to(self.device)
        loss_d1_constraint = torch.norm(torch.mm(d,d.t()) * I - I)
        return 	1e-3 * loss_d1_constraint

   
class D_constraint2(torch.nn.Module):

    def __init__(self, device, beta = 0.01):
        super(D_constraint2, self).__init__()
        self.device = device
        self.beta = beta

    def forward(self, d, dim, n_clusters):
        S = torch.ones(d.shape[0],d.shape[0]).to(self.device)
        zero = torch.zeros(dim, dim)
        for i in range(n_clusters):
            S[i*dim:(i+1)*dim, i*dim:(i+1)*dim] = zero
        loss_d2_constraint = torch.norm(torch.mm(d,d.t()) * S)
        return 1e-3 *  loss_d2_constraint


