from functools import reduce
import math
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F
from entmax import entmax_bisect
from layers import *

class GraphRecommender(nn.Module):
    def __init__(self, opt, num_node, adj, len_session, n_train_sessions):
        super(GraphRecommender, self).__init__()

        self.batch_size = opt.batchSize
        self.num_node = num_node
        self.len_session = len_session

        self.dim = opt.hiddenSize

        self.item_embedding = nn.Embedding(num_node, self.dim,
                                           padding_idx=0)
        self.pos_embedding = nn.Embedding(self.len_session, self.dim)


        self.item_conv = GlobalItemConv(layers=1)
        # 'tmall': opt.w_k = 16  opt.dropout = 0.4
        # 'retailrocket': opt.w_k = 12 opt.dropout = 0.2
        # 'diginetica': opt.w_k = 12  opt.dropout = 0.2
        w_k=16
        dropout=0.4
        self.w_k = w_k
        self.adj = adj
        self.dropout = dropout

        self.n_sessions = n_train_sessions

        # pos attention
        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        
        self.w_f = nn.Linear(2*self.dim, self.dim)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def compute_sess_emb(self, item_seq, hidden, rev_pos=True, attn=True):
        batch_size = hidden.shape[0]
        len = hidden.shape[1]
        mask = torch.unsqueeze((item_seq != 0), -1)

        hs = torch.sum(hidden * mask, -2) / torch.sum(mask, 1)
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = hidden

        if rev_pos:
            pos_emb = self.pos_embedding.weight[:len]
            pos_emb = torch.flip(pos_emb, [0])  # reverse order
            pos_emb = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)
            nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
            nh = torch.tanh(nh)

        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))

        if attn:
            beta = torch.matmul(nh, self.w_2)
            beta = beta * mask
            sess_emb = torch.sum(beta * hidden, 1)
        else:
            sess_emb = torch.sum(nh * hidden, 1)

        return sess_emb

    def get_alpha(self, x=None):
        # x[b,1,d]
        alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  #[b,1,1]
        alpha_global = self.add_value(alpha_global)
        return alpha_global #[b,1,1]

    # 意图： 将value中等于 1 的位置修正为 1.00001，但value中似乎没有等于1的值
    def add_value(self, value):
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        # if (value==1).any():
        #     print("value:", value[value==1])
        return value

    # 这段代码实现了一个全局注意力机制，通过对输入的 target、k 和 v 进行线性变换和非线性激活后，计算得到注意力权重 alpha，然后将其归一化并用于加权求和输入的值向量 v，最终返回加权后的上下文向量 c
    def tglobal_attention(self, target, k, v, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias),self.atten_w0.t())
        alpha = entmax_bisect(alpha, alpha_ent, dim=1)
        c = torch.matmul(alpha.transpose(1, 2), v)
        return c

    def forward(self, global_items, global_inputs, global_alias_inputs, target_embedding, cl=False):
        
        '''
        items: 当前会话不重复物品序列 + 0填充
        alias_input:  原始会话 对应的 不重复物品下标
        input: 原始会话
        '''
        
        items, inputs, alias_inputs = global_items, global_inputs, global_alias_inputs
        graph_item_embs = self.item_conv(self.item_embedding.weight, self.adj)
        hidden = graph_item_embs[items]
        # print("hidden.shape:", hidden.shape)

        # dropout
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        # alias_inputs = alias_inputs.view(-1, alias_inputs.size(1), 1).expand(-1, -1, self.dim)
        # seq_hidden = torch.gather(hidden, dim=1, index=alias_inputs)
        # print("seq_hidden.shape:", seq_hidden.shape)
        # reverse position attention
        # sess_emb = self.compute_sess_emb(inputs, seq_hidden, rev_pos=True, attn=True)

        # weighted L2 normalization: NISER, DSAN, STAN, COTREC
        # select = self.w_k * F.normalize(sess_emb, dim=-1, p=2)
        # len = seq_hidden.shape[1]
        # select = select.unsqueeze(1).expand(select.shape[0], len, select.shape[1]) #[b,s,d]
        select = hidden
        
        # print("select.shape:", select.shape)
        target_emb = self.w_f(target_embedding) # target_emb:(b,1,d)
        # print("target_embedding.shape,target_emb.shape:", target_embedding.shape, target_emb.shape)
        # print("target_emb.shape:", target_emb.shape)
        alpha_line = self.get_alpha(x=target_emb) # alpha_line: (b,1,1)
        
        q = target_emb #[b,1,d]
        k = select #[b,s,d]
        v = select #[b,s,d]
        '''
        Q:查询是用来获取信息的向量，它通常是注意力机制中要进行比较的对象
        K:键是用来匹配查询的向量，它与查询向量进行比较
        V:值是对于查询和键的注意力分数的响应
        '''

        line_c = self.tglobal_attention(q, k, v, alpha_ent=alpha_line) #[b,1,d]
        c = torch.selu(line_c).squeeze() # c进行selu激活，之后去除维度为1的列 c:[b,d]
        '''
        torch.norm(c, dim=-1): 这个函数计算张量 c 沿着最后一个维度（dim=-1，通常是最后一个维度）的范数。
        如果 c 的形状是 [batch_size, output_size]，那么这个操作会计算每个样本向量的范数，结果将是一个形状为 [batch_size] 的张量。
        '''
        # print("c.shape:", c.shape)
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
    
        # graph_item_embs_norm = F.normalize(graph_item_embs, dim=-1, p=2)
        # print("graph_item_embs_norm:", select.shape)
        # scores = torch.matmul(select, graph_item_embs_norm.transpose(1, 0))

        # print("inputs.shape:", inputs.shape)
        # con_loss = torch.Tensor(0)
        # if cl:
        #     con_loss = self.compute_con_loss(batch, select, graph_item_embs_norm)

        return l_c
