import torch

import torch.nn as nn

from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm.batch_norm import BatchNorm
import torch.nn.functional as F


class AttModule(nn.Module):
    
    def __init__(self, hidden_dim, snap_counter=3) -> None:
        super(AttModule, self).__init__()  
        
        self.Q = nn.Linear(hidden_dim, 1)
        
    def forward(self, h1, h2, h3):
        # print(h1.shape, h2.shape, h3.shape)
        context = torch.cat([h1,h2,h3])
        # print(context.shape)
        
        unnormal_weights = self.Q(context)
        unnormal_weights = torch.tanh(unnormal_weights)
        normal_weights = F.softmax(unnormal_weights, dim=0)
        ret = normal_weights[0] * h1 + normal_weights[1] * h2 + normal_weights[2] * h3
        # print(ret, ret.shape)
        # input()
        return ret