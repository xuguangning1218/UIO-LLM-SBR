import math
import torch
import torch.nn.functional as F
from torch import nn
from model.DUIGNN import DUIGNN
from model.IUIGNN import IUIGNN
from model.RES import RES

class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()

    def forward(self, scores, targets):
        log_scores = -scores.log()
        nllloss = log_scores.gather(dim=-1, index=targets.to(torch.int64).unsqueeze(-1))
        return nllloss.mean()

class UIOSBR(nn.Module):

    def __init__(self, opt):
        super(UIOSBR, self).__init__()
        
        
        ###################################### Parameter ####################################################
        self.delta = opt.delta
        self.session_truncated_len = opt.session_truncated_len
        self.emb_dropout = opt.emb_dropout
        self.gnn_dropout = opt.gnn_dropout
        self.use_iuignn = opt.use_iuignn
        self.use_duignn = opt.use_duignn
        self.use_res = opt.use_res
        self.use_seq_item_occur_attn = opt.use_seq_item_occur_attn
        self.iuignn_step = opt.iuignn_step
        self.duignn_step = opt.duignn_step
        self.hidden_size = opt.hiddenSize
        self.batch_size = opt.batchSize
        self.n_node = {
            "diginetica": 43097,
            "Tmall": 40727,
            "Nowplaying": 60416,
            "retailRocket_DSAN": 36968
        }
        #####################################################################################################
        
        
        ###################################### Embedding ####################################################
        # Item embedding
        self.item_emb = nn.Embedding(self.n_node[opt.dataset] + 1, self.hidden_size, padding_idx=0)
        # Item Occurrence embedding
        self.item_occur_emb = nn.Embedding(self.session_truncated_len+1, self.hidden_size)
        # Item Position Embedding
        self.item_pos_emb = nn.Embedding(self.session_truncated_len+1, self.hidden_size, padding_idx=0)
        # init embedding
        stdv = 1.0 / math.sqrt(self.hidden_size)  
        self.item_emb.weight.data.uniform_(-stdv, stdv)  
        self.item_occur_emb.weight.data.uniform_(-stdv, stdv)  
        self.item_pos_emb.weight.data.uniform_(-stdv, stdv)
        #####################################################################################################
        
        
        ###################################### IUI-GNN #######################################################
        # GNN layer in IUI-GNN
        self.iuignn = IUIGNN(self.hidden_size, self.iuignn_step)
        #####################################################################################################
        
        ###################################### DUI-GNN #######################################################
        # GNN layer in DUI-GNN
        self.duignn = DUIGNN(self.hidden_size, self.use_duignn, self.duignn_step)
        self.linear_trans = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        #####################################################################################################
        
        #################################### Review-and-Explore Strategy #####################################
        self.linear_review = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_explore = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.review_attn_value_generator = nn.Parameter(torch.rand(size=[1 * self.hidden_size,1]))
        self.explore_attn_value_generator = nn.Parameter(torch.rand(size=[1 * self.hidden_size,1]))
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.res = RES(self.hidden_size,self.delta)
        #####################################################################################################
        
    
    def forward(self, alias_item_id_in_session, adj_out, adj_in, unique_item_id_in_session, item_id_in_session):
        
        # use mask session for aligment
        session_masker = item_id_in_session.bool().unsqueeze(-1)
        session_len = session_masker.sum(-2)
        
        # item embedding in session
        item_emb_in_session = self.item_emb(item_id_in_session)
        item_emb_in_session = F.dropout(item_emb_in_session, self.emb_dropout, training=self.training)


        ###################################### DUI-GNN ##########################################################################
        # unique item embeding in session
        unique_item_emb_in_session = self.item_emb(unique_item_id_in_session)
        unique_item_emb_in_session = F.dropout(unique_item_emb_in_session, self.emb_dropout, training=self.training)
        
        # DUI-GNN layer
        # merge_occur_item_embs \in R^{ b \times n \times 100} 
        merge_occur_item_embs = self.duignn(
            adj_out, adj_in, 
            unique_item_emb_in_session, item_id_in_session, 
            alias_item_id_in_session, unique_item_id_in_session, self.item_occur_emb)
        # alias_item_id_in_session is sequential item embedding in session
        merge_occur_item_embs = merge_occur_item_embs.gather(
            dim=-2, index=alias_item_id_in_session.unsqueeze(-1).expand(-1, -1, merge_occur_item_embs.size(-1)))
        
        # Sequential Item Occurrence Attention
        item_occur_emb_in_session, _ = self.get_occur_item_embs(item_id_in_session)
        if self.use_seq_item_occur_attn:
            normal_item_occur_emb_in_session = F.normalize(item_occur_emb_in_session, dim=-1)
            attn_values = self.linear_trans(merge_occur_item_embs).sigmoid()
            merge_occur_item_embs = torch.einsum(
                'bth,bth,btg->btg',
                attn_values, normal_item_occur_emb_in_session, merge_occur_item_embs)
        ########################################################################################################################
        
        
        ###################################### Merging IUI-GNN and DUI-GNN #######################################################
        if self.use_iuignn:
            pos_emb = self.iuignn(item_id_in_session, self.item_pos_emb)
            merge_duignn_iuignn = pos_emb + merge_occur_item_embs
        else:
            merge_duignn_iuignn = merge_occur_item_embs
        ########################################################################################################################
        
        
        ######################################## Review-and-Explore Strategy #####################################################
        item_emb_in_session = F.normalize(item_emb_in_session, p=2, dim=-1)
        
        # weight for merging explore and review
        gru_occur_hidden, _ = self.gru(item_occur_emb_in_session+item_emb_in_session.detach())
        gru_occur_hidden = gru_occur_hidden * session_masker

        # Items Characteristics Representation 1 for reviewing
        review_attn_value_generator = F.normalize(self.review_attn_value_generator, dim=0)
        merge_duignn_iuignn_alpha = self.linear_review(merge_duignn_iuignn)
        review_attn_value = merge_duignn_iuignn_alpha.sigmoid().matmul(review_attn_value_generator) * session_masker
        review_sess_emb = (review_attn_value * session_masker).transpose(1, 2).bmm(item_emb_in_session).squeeze(1)
        review_score = self.compute_scores(review_sess_emb, self.item_emb.weight)
 
        # Items Characteristics Representation 2 for exploring
        explore_attn_value_generator = F.normalize(self.explore_attn_value_generator, dim=0)
        merge_duignn_iuignn_gama = self.linear_explore(merge_duignn_iuignn)
        explore_attn_value = merge_duignn_iuignn_gama.sigmoid().matmul(explore_attn_value_generator) * session_masker
        explore_sess_emb = (explore_attn_value * session_masker).transpose(1, 2).bmm(item_emb_in_session).squeeze(1)
        explore_score = self.compute_scores(explore_sess_emb, self.item_emb.weight)

        if self.use_res:
            prob = self.res(item_id_in_session.size(0), self.item_emb.weight.size(0),item_id_in_session.device,unique_item_id_in_session,review_score,explore_score,gru_occur_hidden,session_len)
        else:
            prob = review_score.mul(self.delta).softmax(-1)
        ########################################################################################################################
        
        return prob
    
    def get_occur_item_embs(self, item_id_in_session):
        item_id_in_session = item_id_in_session.unsqueeze(-1)  # b x t x 1
        session_masker = item_id_in_session.bool()  # b x t x 1
        ######################################################################################
        # if A_i == A_j the result is 0 otherwise the result is not 0.
        relation_matrix = item_id_in_session - item_id_in_session.transpose(1, 2)  # b x t x t
        ######################################################################################
        relation_matrix = relation_matrix.bool()  # b x t x t
        # keep the same item index 
        relation_matrix = (~relation_matrix).long().tril()
        relation_matrix = relation_matrix * session_masker * session_masker.transpose(1, 2)
        seq_occur_count = relation_matrix.sum(2)  # b x t
        return self.item_occur_emb(seq_occur_count), seq_occur_count  # b x t x h
    
    def compute_scores(self, x, A):
        # A [unique_item_id_in_session x hidden]
        x = F.normalize(x, p=2, dim=1)
        A = F.normalize(A, p=2, dim=1)
        scores = torch.matmul(x, A.t())  # batch x unique_item_id_in_session
        return scores

