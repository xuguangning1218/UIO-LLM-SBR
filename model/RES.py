import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RES(nn.Module):
    def __init__(self, hidden_size, delta):
        super(RES, self).__init__()
        self.delta = delta
        self.linear_gru = nn.Linear(2 * hidden_size, hidden_size, bias=False)
        self.linear_prob_condition = nn.Parameter(torch.rand(size=[2, hidden_size]))
        
    def forward(self, batch_size, item_size, tensor_device,unique_item_id_in_session,review_score,explore_score,gru_occur_hidden,session_len):
        # mask_shown_up for exploring
        # mask_unshown_uper for reviewing
        zeros = torch.zeros(size=[batch_size, item_size], dtype=torch.long, device=tensor_device)
        mask_shown_up, mask_unshown_up = self.get_items_appearance_masker(unique_item_id_in_session, zeros)
        
        review_score = review_score.masked_fill(mask_unshown_up, float('-1')).mul(self.delta)
        review_prob = review_score.softmax(-1)
        explore_score = explore_score.masked_fill(mask_shown_up, float('-1')).mul(self.delta)
        explore_prob = explore_score.softmax(-1)

        # User Preference Representation h_u in R^{b \times 100}
        user_preference = self.linear_gru(gru_occur_hidden.sum(1) / session_len)
        weights_for_explore_review = self.compute_scores(user_preference, self.linear_prob_condition).mul(2)  # [b,2]
        weights_for_explore_review = weights_for_explore_review.softmax(-1)
        
        prob = torch.stack([review_prob,explore_prob], dim=-1).bmm(weights_for_explore_review.unsqueeze(-1)).squeeze(-1)     
        return prob

    def get_items_appearance_masker(self, unique_item_id_in_session, zeros):
        mask_shown_up = zeros.scatter(dim=-1, index=unique_item_id_in_session, value=1)
        mask_shown_up = mask_shown_up.bool()
        mask_unshown_up = ~mask_shown_up
        return mask_shown_up, mask_unshown_up

    def compute_scores(self, x, A):
        # A [unique_item_id_in_session x hidden]
        x = F.normalize(x, p=2, dim=1)
        A = F.normalize(A, p=2, dim=1)
        scores = torch.matmul(x, A.t())  # batch x unique_item_id_in_session
        return scores