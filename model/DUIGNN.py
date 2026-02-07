import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class BiGNNLayer(nn.Module):
    def __init__(self, hidden_size):
        super(BiGNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.Tensor(1 * self.hidden_size))
        self.linear_edge_in = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=False)
        self.linear_edge_out = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=False)

    def forward(self, adj_out, adj_in, hidden, u):
        h_out = self.linear_edge_out(hidden)
        h_in = self.linear_edge_in(hidden)
        result_out = torch.matmul(adj_out, h_out + u) + self.bias
        result_in = torch.matmul(adj_in.transpose(1, 2), h_in + u)  
        result = result_in + result_out
        return result

class DUIGNN(nn.Module):
    def __init__(self, hidden_size, use_duignn, step=1):
        super(DUIGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.use_duignn = use_duignn
        self.gnn_layers = nn.ModuleList([
            BiGNNLayer(hidden_size) for _ in range(step)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_occur_indices(self, item_id_in_session, alias_item_id_in_session, max_n_node):
        item_id_in_session = item_id_in_session.unsqueeze(-1)  # b x t x 1
        seq_mask = item_id_in_session.bool()  # b x t x 1

        relation_matrix = item_id_in_session - item_id_in_session.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).long()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(1, 2)
        seq_occur_count = relation_matrix.sum(2)  # b x t
        res = torch.zeros([item_id_in_session.size(0), max_n_node], dtype=torch.long, device=item_id_in_session.device)
        res = res.scatter(dim=-1, index=alias_item_id_in_session, src=seq_occur_count)
        return res

    def forward(self, adj_out, adj_in, unique_item_emb_in_session, item_id_in_session, alias_item_id_in_session, unique_item_id_in_session,item_occur_emb):
        '''
        A: b x t x t
        unique_item_emb_in_session: b x t x h
        '''
        if self.use_duignn:
            indices = self.get_occur_indices(item_id_in_session, alias_item_id_in_session, unique_item_id_in_session.size(-1))
            item_occur_emb_in_session = item_occur_emb(indices)  # b x t x h
            item_occur_emb_in_session = F.normalize(item_occur_emb_in_session, dim=-1)
        else:
            item_occur_emb_in_session = torch.zeros_like(unique_item_emb_in_session, device=unique_item_emb_in_session.device)

        for i in range(self.step):
            unique_item_emb_in_session = self.gnn_layers[i](adj_out, adj_in, unique_item_emb_in_session, item_occur_emb_in_session)
        return unique_item_emb_in_session
