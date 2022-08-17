# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_batch

class MLP(nn.Sequential):
    def __init__(self, hidden_dim, num_layers, dropout=0.5):
        def build_block(input_dim, output_dim):
            return [
                nn.Linear(input_dim, output_dim),
                nn.BatchNorm1d(output_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
        
        m = build_block(hidden_dim, 2 * hidden_dim)
        for i in range(1, num_layers - 1):
            m += build_block(2 * hidden_dim, 2 * hidden_dim)
        m.append(nn.Linear(2 * hidden_dim, hidden_dim))
        
        super().__init__(*m)

class GINConv(MessagePassing):
    def __init__(self, hidden_dim, num_layers):
        super().__init__()
        self.mlp = MLP(hidden_dim, num_layers, 0.0)
    
    def forward(self, x, edge_index, size=None):
        out = self.propagate(edge_index, x=(x, x), size=size)
        out = x + self.mlp(out)
        return out
    
    def message(self, x_j):
        return x_j

class SubExtractor(nn.Module):
    def __init__(self, hidden_dim, num_clusters, residual=False):
        super().__init__()
        
        self.Q = nn.Parameter(torch.Tensor(1, num_clusters, hidden_dim))
        nn.init.xavier_uniform_(self.Q)
        
        self.W_Q = nn.Linear(hidden_dim, hidden_dim)
        self.W_K = nn.Linear(hidden_dim, hidden_dim)
        self.W_V = nn.Linear(hidden_dim, hidden_dim)
        self.W_O = nn.Linear(hidden_dim, hidden_dim)

        self.residual = residual
    
    def forward(self, x, batch):
        K = self.W_K(x)
        V = self.W_V(x)
        
        K, mask = to_dense_batch(K, batch)
        # mask: (batch_size, max_num_nodes)
        V, _ = to_dense_batch(V, batch)
        
        attn_mask = (~mask).float().unsqueeze(1)
        attn_mask = attn_mask * (-1e9)
        
        Q = self.Q.tile(K.size(0), 1, 1)
        Q = self.W_Q(Q)
        
        A = Q @ K.transpose(-1, -2) / (Q.size(-1) ** 0.5)
        A = A + attn_mask
        A = A.softmax(dim=-2)
        # (batch_size, num_clusters, max_num_nodes)
        
        out = Q + A @ V
        
        if self.residual:
            out = out + self.W_O(out).relu()
        else:
            out = self.W_O(out).relu()
        
        return out, A.detach().argmax(dim=-2), mask
