import torch
import numpy as np
# from utils.geo_data import Data
from torch.utils.data import Dataset

class SBRDataSet(Dataset):
    def __init__(self, data, session_truncated_len):
        super(SBRDataSet, self).__init__()
        inputs, max_len = self.data_masks(data[0], max_len=session_truncated_len)
        self.inputs = np.asarray(inputs)
        self.targets = np.asarray(data[1])
        self.length = len(data[0])
        # self.inputs = np.asarray(inputs)[:64]
        # self.targets = np.asarray(data[1])[:64]
        # self.length = 64
        
        self.max_len = max_len

    def data_masks(self, all_usr_pois, item_tail=None, max_len=100):
        if item_tail is None:
            item_tail = [0]
        new_all_usr_pois = []
        for upois in all_usr_pois:
            if len(upois) > max_len:
                upois = upois[-max_len:]
            new_all_usr_pois.append(upois)
        us_lens = [len(upois) for upois in new_all_usr_pois]
        len_max = max(us_lens) + 1  # note: the last signal must be zero for adapting to graph construction
        us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(new_all_usr_pois, us_lens)]
        return us_pois, len_max
    
    def __getitem__(self, index):

        inputs, target = self.inputs[index], self.targets[index]
        max_n_node = self.max_len

        inputs = torch.tensor(inputs, dtype=torch.long)
        node = torch.unique(inputs)
        padding = torch.zeros(size=[max_n_node - len(node)], dtype=torch.long, device=inputs.device)
        items = torch.cat([node, padding])

        relation_matrix = (node.unsqueeze(-1) - inputs.unsqueeze(-1).t()).abs()
        alias_inputs = torch.argmin(relation_matrix, dim=0)
        adj_out, adj_in = self.construct_graph(alias_inputs, max_n_node, inputs.device)

        freq_index = torch.where(inputs==target)
        freq_target = torch.ones(size=(),dtype=torch.long)
        if freq_index[0].size(0)==0:
            freq_target = freq_target*0
        else:
            seq_freq = self.get_freq_position(inputs.unsqueeze(-1),inputs.unsqueeze(-1).bool())
            freq_target = freq_target*seq_freq[freq_index[0][-1]]

        return [alias_inputs, adj_out, adj_in, items, inputs, torch.tensor(target, device=inputs.device), freq_target]

    def __len__(self):
        return self.length
    
    def get_freq_position(self, seq_inputs, seq_mask):
        '''
        seq_inputs: [t,1]
        '''
        relation_matrix = seq_inputs - seq_inputs.transpose(-2, -1)  # t x t
        relation_matrix = relation_matrix.bool()  # t x t
        relation_matrix = (~relation_matrix).long().tril()
        relation_matrix = relation_matrix * seq_mask * seq_mask.transpose(-2, -1)
        seq_occur_count = relation_matrix.sum(-1).clamp(min=0,max=301)  # t
        return seq_occur_count # [t]

    def construct_graph(self, alias_input, max_n_node, device):
        alias_input = torch.cat([alias_input, torch.zeros([1], dtype=torch.long, device=device)])
        u_A = torch.zeros((max_n_node, max_n_node), dtype=torch.float, device=device)
        coordinte = torch.arange(alias_input.size(0), device=device)
        alias_input_masked_with_coo = alias_input.bool().long() * coordinte
        first_zero_index = alias_input_masked_with_coo.argmax() + 1

        u_A_xindex = alias_input[:first_zero_index - 1]
        u_A_yindex = alias_input[1:first_zero_index]

        if u_A_xindex.size(0) > 0:
            value = torch.ones_like(u_A_xindex, dtype=torch.float)
            u_A.index_put_([u_A_xindex, u_A_yindex], value, accumulate=False)
            # note: weight is different from srgnn when accumulate=True
        u_A = u_A + torch.eye(max_n_node)

        u_sum_out = torch.sum(u_A, -1)
        u_sum_out[torch.where(u_sum_out == 0)] = 1
        u_A_out = torch.divide(u_A.t(), u_sum_out)

        u_sum_in = torch.sum(u_A, -2, keepdim=True)
        u_sum_in[torch.where(u_sum_in == 0)] = 1
        u_sum_in = torch.divide(u_A, u_sum_in)

        return u_A_out.t(), u_sum_in
