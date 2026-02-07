import math

import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNLayer(nn.Module):
    def __init__(self, hidden_size):
        super(GNNLayer, self).__init__()
        self.hidden_size = hidden_size
        self.bias = nn.Parameter(torch.Tensor(1 * self.hidden_size))
        self.linear_edge = nn.Linear(1 * self.hidden_size, 1 * self.hidden_size, bias=False)

    def forward(self, adj, hidden):
        h = self.linear_edge(hidden)
        result = torch.matmul(adj.transpose(1, 2), h) + self.bias
        return result

class IUIGNN(nn.Module):
    def __init__(self, hidden_size, step=1):
        super(IUIGNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_size) for _ in range(step)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def get_reversed_item_pos_emb(self, session_masker, item_pos_emb):
        length = session_masker.sum(1, keepdim=True)
        coo = torch.arange(session_masker.size(1), device=session_masker.device).unsqueeze(0).expand(session_masker.size(0), -1)
        coo = coo * session_masker
        pos_index = (length - coo) * session_masker
        return item_pos_emb(pos_index)

    def get_forward_item_pos_emb(self, session_masker, item_pos_emb):
        coo = torch.arange(session_masker.size(1), device=session_masker.device).unsqueeze(0).expand(session_masker.size(0), -1)
        pos_index = coo * session_masker
        return item_pos_emb(pos_index)

    def construct_pos_adj(self, item_id_in_session):
        item_id_in_session = item_id_in_session.unsqueeze(-1)  # b x t x 1
        session_masker = item_id_in_session.bool()  # b x t x 1

        relation_matrix = item_id_in_session - item_id_in_session.transpose(1, 2)  # b x t x t
        relation_matrix = relation_matrix.bool()  # b x t x t
        relation_matrix = (~relation_matrix).float().tril(diagonal=0)
        relation_matrix = relation_matrix * session_masker * session_masker.transpose(1, 2)
        
        norm = relation_matrix.sum(-2, keepdim=True)  # b x t x 1
        norm[torch.where(norm == 0)] = 1.
        A = relation_matrix / norm
        return A # b x t x t

    def forward(self, item_id_in_session, item_pos_emb):
        session_masker = item_id_in_session.bool()

        item_pos_emb = self.get_reversed_item_pos_emb(session_masker, item_pos_emb)

        return item_pos_emb  # b x t x h
