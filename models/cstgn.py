

import torch

import torch.nn as nn

from torch_geometric.nn.conv import GCNConv
from torch_geometric.nn.norm.batch_norm import BatchNorm
import torch.nn.functional as F

from .att import AttModule
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CSTGN(nn.Module):
    
    def __init__(self, num_features, hidden_dim) -> None:
        super(CSTGN, self).__init__()        
        
        self.query_gcn_layer = GCNConv(num_features, hidden_dim)
        self.graph_gcn_layer = GCNConv(num_features, hidden_dim)
        
        self.batch1g = BatchNorm(hidden_dim)
        self.batch1q = BatchNorm(hidden_dim)
        
        self.query_gcn_layer2 = GCNConv(hidden_dim, hidden_dim)
        self.graph_gcn_layer2 = GCNConv(hidden_dim, hidden_dim)
        
        self.batch2g = BatchNorm(hidden_dim)
        self.batch2q = BatchNorm(hidden_dim)
        
        self.query_gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        self.graph_gru_layer = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)
        
        self.graph_att = AttModule(hidden_dim, 3)
        
        self.hidden_dim = hidden_dim
        
        self.linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, data, query, g_hidden=None, q_hidden=None, g_past=None, g_pastpast=None):
        x = data.x
        q = query
        edge_index = data.edge_index
        
        # Snapshot 0
        if q_hidden is None:
            g_hidden = self.init_hidden(self.hidden_dim)
            q_hidden = self.init_hidden(self.hidden_dim)
        
        # Snapshot 1
        if g_past is None:
            g_past = self.init_hidden(self.hidden_dim)
        # Snapshot 1, 2
        if g_pastpast is None:
            g_pastpast = self.init_hidden(self.hidden_dim)

        x = self.graph_gcn_layer(x, edge_index)
        x = self.batch1g(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        
        
        q = self.query_gcn_layer(q, edge_index)
        q = self.batch1q(q)
        q = F.relu(q)
        q = F.dropout(q, training=self.training)
        
        g_hidden = self.graph_att(g_hidden, g_past, g_pastpast)
        x, g_hidden = self.graph_gru_layer(x, g_hidden)
        q_hidden += g_hidden
        q, q_hidden = self.query_gru_layer(q, q_hidden)
        

        x = self.graph_gcn_layer2(x, edge_index)
        x = self.batch2g(x)
        q = self.query_gcn_layer2(q, edge_index)
        q = self.batch2q(q)

        output = F.relu(x) + F.relu(q)
        
        output = self.linear(output)  
        
        return torch.sigmoid(output).squeeze(), g_hidden, q_hidden, 

        
        
    
    def init_hidden(self, dim):
        return torch.zeros((1,dim)).to(device)


