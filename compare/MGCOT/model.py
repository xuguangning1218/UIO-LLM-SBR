#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import datetime
import math
from math import sqrt
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
import torch.nn.init as init
from numba import jit
from entmax import entmax_bisect
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from contrastive import ContrastiveLearningModel


'''
该段代码的中心思想是实现了 Batch Normalization 的功能，
通过计算输入张量 x 在特征维度上的均值和方差，对其进行标准化处理，
然后通过学习的参数 self.weight 和 self.bias 进行线性变换，最终输出标准化和线性变换后的结果。
'''
class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


# 学习会话中每个节点的信息，结合GNN和GRU性质，对应论文公式。
class GNN(Module):
    def __init__(self, hidden_size, step=1):
        super(GNN, self).__init__()
        self.step = step
        self.hidden_size = hidden_size * 2
        self.input_size = self.hidden_size * 2
        self.gate_size = 3 * self.hidden_size
        self.w_ih = Parameter(torch.Tensor(self.gate_size, self.input_size)) # (6d, 4d)
        self.w_hh = Parameter(torch.Tensor(self.gate_size, self.hidden_size)) # (6d, 2d)
        self.b_ih = Parameter(torch.Tensor(self.gate_size))
        self.b_hh = Parameter(torch.Tensor(self.gate_size))
        self.b_iah = Parameter(torch.Tensor(self.hidden_size))
        self.b_oah = Parameter(torch.Tensor(self.hidden_size))

        self.linear_edge_in = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_out = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.linear_edge_f = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def GNNCell(self, A, hidden):
        input_in = torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)) + self.b_iah # input_in: (b,s,2d)
        # A[:, :, :A.shape[1]]: [b,s,s] self.linear_edge_in(hidden):[b,s,2d] self.b_iah: 2d input_in: [b,s,2d]
        # print("input_in.shape:", input_in.shape) # [b,s,2d]
        # print("A[:, :, :A.shape[1]]:", A[:, :, :A.shape[1]].shape)
        # print("self.linear_edge_in(hidden):", self.linear_edge_in(hidden).shape)
        # print("torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)):", torch.matmul(A[:, :, :A.shape[1]], self.linear_edge_in(hidden)).shape)
        # print("self.b_iah:", self.b_iah.shape)
        
        # input_in = A_in * linear_edge+b_iah
        # A:100*8*16
        # hidden = 100*8*100
        # b_iah 10
        input_out = torch.matmul(A[:, :, A.shape[1]: 2 * A.shape[1]], self.linear_edge_out(hidden)) + self.b_oah
        inputs = torch.cat([input_in, input_out], 2) # inputs: (b,s,4d) inputs=>  p_k^l
        # print("inputs.shape:", inputs.shape)
        gi = F.linear(inputs, self.w_ih, self.b_ih) # gi: (b，s, 6d） # self.w_ih 是权重张量，形状为 [out_features, in_features]
        # g_i => W_z*p_k^l
        gh = F.linear(hidden, self.w_hh, self.b_hh)  # gh: (b,s,6d)
        # g_h => U_z*x_k^l
        # print("gi,gh:", gi.shape, gh.shape)
        i_r, i_i, i_n = gi.chunk(3, 2)  
        # gi.chunk(3,2)：沿着gi的第二维切分为3块 (b,s,2d)
        # print("i_r,i_i,i_n.shape:", i_r.shape, i_i.shape, i_n.shape) 
        h_r, h_i, h_n = gh.chunk(3, 2)
        resetgate = torch.sigmoid(i_r + h_r)  # (b,s,2d) # resetgate => r_k^l
        inputgate = torch.sigmoid(i_i + h_i) # inputgate => z_k^l
        newgate = torch.tanh(i_n + resetgate * h_n)  # *： element-wise product # newgate=>~s(x_k^l)
        hy = newgate + inputgate * (hidden - newgate) # 是否和论文公式相符？
        # hy = (1-inputgate)*hidden + inputgate*newgate
        # print("hy:", hy.shape) (b,s,2d)
        return hy

    def forward(self, A, hidden):  
        for i in range(self.step): 
            hidden = self.GNNCell(A, hidden)
        return hidden


class FindNeighbors(Module):
    def __init__(self, hidden_size, opt):
        super(FindNeighbors, self).__init__()
        self.hidden_size = hidden_size
        self.neighbor_n = opt.neighbor_n # Diginetica:3; Tmall: 7; Nowplaying: 4
        print("self.neighbor:", self.neighbor_n)
        self.dropout40 = nn.Dropout(0.40)

    # 计算会话间余弦相似度
    def compute_sim(self, sess_emb):
        # print("sess_emb:", sess_emb.shape) (b,d)
        fenzi = torch.matmul(sess_emb, sess_emb.permute(1, 0)) 
        fenmu_l = torch.sum(sess_emb * sess_emb + 0.000001, 1)
        fenmu_l = torch.sqrt(fenmu_l).unsqueeze(1)
        fenmu = torch.matmul(fenmu_l, fenmu_l.permute(1, 0))
        cos_sim = fenzi / fenmu 
        # 将张量 cos_sim 沿着最后一个维度进行 softmax 归一化
        cos_sim = nn.Softmax(dim=-1)(cos_sim)
        return cos_sim

    '''
    该段代码的中心思想是根据输入的会话特征向量 sess_emb，计算每个会话向量与其它会话向量之间的余弦相似度，
    然后根据这些相似度选择每个会话的最相似的前 k_v 个会话。
    接着，利用这些最相似的会话向量，加权求和得到每个会话的邻居向量表示。
    最后，通过 dropout 操作来增强模型的泛化能力，并将处理后的向量作为输出返回。
    '''
    def forward(self, sess_emb):
        k_v = self.neighbor_n 
        cos_sim = self.compute_sim(sess_emb) 
        # print("cos_sim:", cos_sim.size()[0]) cos_sim.size()[0]：批次会话数
        if cos_sim.size()[0] < k_v:
            k_v = cos_sim.size()[0]
        cos_topk, topk_indice = torch.topk(cos_sim, k=k_v, dim=1) # cos_topk: [b, k_v]
        cos_topk = nn.Softmax(dim=-1)(cos_topk)
        sess_topk = sess_emb[topk_indice] # sess_topk: (b, k_v, hidden)
        # print("sess_topk:", sess_topk.shape)

        cos_sim = cos_topk.unsqueeze(2).expand(cos_topk.size()[0], cos_topk.size()[1], self.hidden_size)

        neighbor_sess = torch.sum(cos_sim * sess_topk, 1)
        neighbor_sess = self.dropout40(neighbor_sess)  # [b,d] 加权求和得到每个会话的邻居向量表示
        return neighbor_sess



class RelationGAT(Module):
    def __init__(self, batch_size, hidden_size=100):
        super(RelationGAT, self).__init__()
        self.batch_size = batch_size
        self.dim = hidden_size
        self.w_f = nn.Linear(2*hidden_size, hidden_size)
        self.alpha_w = nn.Linear(self.dim, 1)
        self.atten_w0 = nn.Parameter(torch.Tensor(1, self.dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(self.dim, self.dim))
        self.atten_bias = nn.Parameter(torch.Tensor(self.dim))

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

    # 将输入的目标嵌入（线性、非线性激活）、物品嵌入（GCN）经过注意力机制等一系列计算操作后，得到每个会话归一化后的向量表示。
    def forward(self, item_embedding, items, A, D, target_embedding):
        '''
        item_embedding: (n_node,hidden_size)
        items: (batch_size, original_session_len)
        A: (batch_size, batch_size)
        D: (batch_size, batch_size)
        target_embedding: (batch_size, 1, 2*hidden_size)
        '''
        seq_h = []
        for i in torch.arange(items.shape[0]): # 如果按照物品序列顺序输入，对结果是否有影响？
            seq_h.append(torch.index_select(item_embedding, 0, items[i]))  # [b,s,d] items[i]:第i个会话包含的物品集合 torch.index_select（input, dim, index）
        
        # seq_h1 = trans_to_cuda(torch.tensor([item.cpu().detach().numpy() for item in seq_h])) # seq_h1: (b,s,d)
        seq_h_array = np.array([item.cpu().detach().numpy() for item in seq_h])
        seq_h1 = torch.tensor(seq_h_array, dtype=torch.float32, device='cuda')
        # print("seq_h1.shape:", seq_h1.shape)
        len = seq_h1.shape[1]
        relation_emb_gcn = torch.sum(seq_h1, 1) #[b,d]
        DA = torch.mm(D, A).float() #[b,b]
        relation_emb_gcn = torch.mm(DA, relation_emb_gcn) #relation_emb_gcn: [b,d]
        
        # relation_emb_gcn = relation_emb_gcn.unsqueeze(1).expand(relation_emb_gcn.shape[0], len, relation_emb_gcn.shape[1]) #[b,s,d]

        # target_emb = self.w_f(target_embedding) # target_emb:(b,1,d)
        # # print("target_embedding.shape,target_emb.shape:", target_embedding.shape, target_emb.shape)
        
        # alpha_line = self.get_alpha(x=target_emb) # alpha_line: (b,1,1)
        
        # q = target_emb #[b,1,d]
        # k = relation_emb_gcn #[b,1,d]
        # v = relation_emb_gcn #[b,1,d]
        # '''
        # Q:查询是用来获取信息的向量，它通常是注意力机制中要进行比较的对象
        # K:键是用来匹配查询的向量，它与查询向量进行比较
        # V:值是对于查询和键的注意力分数的响应
        # '''

        # line_c = self.tglobal_attention(q, k, v, alpha_ent=alpha_line) #[b,1,d]
        # c = torch.selu(line_c).squeeze() # c进行selu激活，之后去除维度为1的列 c:[b,d]
        # '''
        # torch.norm(c, dim=-1): 这个函数计算张量 c 沿着最后一个维度（dim=-1，通常是最后一个维度）的范数。
        # 如果 c 的形状是 [batch_size, output_size]，那么这个操作会计算每个样本向量的范数，结果将是一个形状为 [batch_size] 的张量。
        # '''
        # # print("c.shape:", c.shape)
        # l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        # # print("l_c:", l_c.shape)

        # return l_c #[b,d]
    
        relation_emb_gcn = (relation_emb_gcn / torch.norm(relation_emb_gcn, dim=-1).unsqueeze(1))
        return relation_emb_gcn





class SessionGraph(Module):
    def __init__(self, opt, n_node, graphRecommender):
        super(SessionGraph, self).__init__()
        self.dataset = opt.dataset
        self.hidden_size = opt.hiddenSize
        self.n_node = n_node
        self.batch_size = opt.batchSize
        self.nonhybrid = opt.nonhybrid
        self.embedding = nn.Embedding(self.n_node, self.hidden_size, padding_idx=0, max_norm=1.5)
        self.pos_embedding = nn.Embedding(300, self.hidden_size, padding_idx=0, max_norm=1.5) # 300表示嵌入层能够处理的不同位置 ID 的总数量。
        # 填充索引（padding_idx=0）：用来处理序列中的填充位置，避免它们对模型计算的影响。最大范数（max_norm=1.5）：用来约束嵌入向量的范数，防止嵌入向量变得过大，从而稳定训练过程。
        self.gnn = GNN(self.hidden_size, step=opt.step)


        # Sparse Graph Attention
        self.is_dropout = True
        self.w = 20
        dim = self.hidden_size * 2
        self.dim = dim
        self.LN = nn.LayerNorm(dim) # 用于进行 Layer Normalization（层归一化）操作。这个操作对于神经网络模型中的每个样本的每个特征维度进行归一化处理
        self.LN2 = nn.LayerNorm(self.hidden_size)
        self.dropout = nn.Dropout(0.2)
        self.activate = F.relu
        self.atten_w0 = nn.Parameter(torch.Tensor(1, dim))
        self.atten_w1 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_w2 = nn.Parameter(torch.Tensor(dim, dim))
        self.atten_bias = nn.Parameter(torch.Tensor(dim))
        self.attention_mlp = nn.Linear(dim, dim)
        self.alpha_w = nn.Linear(dim, 1)
        self.self_atten_w1 = nn.Linear(dim, dim)
        self.self_atten_w2 = nn.Linear(dim, dim)
        self.linear2_1 = nn.Linear(2*dim, dim, bias=True)

        #Multi
        self.num_attention_heads = opt.num_attention_heads
        self.attention_head_size = int(dim / self.num_attention_heads)
        self.multi_alpha_w = nn.Linear(self.attention_head_size, 1) # self.attention_head_size：输入特征

        # Neighbor
        self.FindNeighbor = FindNeighbors(self.hidden_size, opt)
        self.w_ne = opt.w_ne
        self.gama = opt.gama

        # Relation Conv
        self.RelationGraph = RelationGAT(self.batch_size, self.hidden_size)
        self.w_f = nn.Linear(4 * self.hidden_size, self.hidden_size)
        self.linear_one = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_two = nn.Linear(2 * self.hidden_size, 2 * self.hidden_size, bias=True)
        self.linear_three = nn.Linear(2 * self.hidden_size, 1, bias=False)
        self.linear_transform = nn.Linear(self.hidden_size * 2, self.hidden_size, bias=True)
        self.LayerNorm = LayerNorm(2*self.hidden_size, eps=1e-12)
        '''
            LayerNorm: 创建了一个LayerNorm对象，它将应用于输入张量的最后一个维度（通常是特征维度）上进行归一化。
            当计算归一化因子时，会加上 eps，例如标准差的计算为 sqrt(variance + eps)，这确保了即使方差很小，也不会导致除以零的错误。
        '''
        self.dropout = nn.Dropout(0.2)
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc) # 调整学习率
        self.reset_parameters()
        
        self.graphRecommender = graphRecommender
        
        self.ContrastiveLearningModel = ContrastiveLearningModel(0.5)

# 对模型的参数进行初始化，使得参数值在训练开始时处于一个合理的范围。
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size) #  Xavier 初始化方法（也称为 Glorot 初始化），其目标是保持前向传播过程中信号的方差恒定。
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv) 
            # weight.data.uniform_(-stdv, stdv): 对每个参数张量的 data 属性（即参数值）进行初始化。uniform_(-stdv, stdv) 方法将参数值从均匀分布 [−stdv,stdv] 中随机采样。


# 将批次会话中的物品：将其物品嵌入和位置嵌入（物品升序而非点击序列）在最后一维上拼接 
# 位置信息根据物品所在的下标进行嵌入表示吗？squence是非重复物品升序集合，这样的话位置嵌入和物品嵌入不是一样的吗？如何体现位置信息？？
# 如果 sequence 是一个升序非重复物品集合的序列，那么 position_ids 确实只是简单地序列的索引，而不是物品在实际会话中的位置信息？？
    def add_position_embedding(self, sequence):

        batch_size = sequence.shape[0]  # b
        len = sequence.shape[1]  # s

        position_ids = torch.arange(len, dtype=torch.long, device=sequence.device)  # [s,]
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # [b,s]
        position_embeddings = self.pos_embedding(position_ids)  # [b,s,d]
        item_embeddings = self.embedding(sequence)

        # sequence_emb = self.linear_transform(torch.cat((item_embeddings, position_embeddings), -1))
        # sequence_emb = item_embeddings + position_embeddings
        
        sequence_emb = torch.cat((item_embeddings, position_embeddings), -1) # sequence_emb(b,s,2d)
        # print("squence_emb.shape:", sequence_emb.shape)
        sequence_emb = self.LayerNorm(sequence_emb)
        # sequence_emb = self.dropout(sequence_emb)

        return sequence_emb


    def get_alpha(self, x=None, seq_len=70, number=None):  # x[b,1,d], seq = len为每个会话序列中最后一个元素
        if number == 0:
            alpha_ent = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_ent = self.add_value(alpha_ent).unsqueeze(1)  # [b,1,1]
            alpha_ent = alpha_ent.expand(-1, seq_len, -1)  # [b,s+1,1]
            return alpha_ent
        if number == 1:  # x[b,1,d]
            alpha_global = torch.sigmoid(self.alpha_w(x)) + 1  # [b,1,1]
            alpha_global = self.add_value(alpha_global)
            return alpha_global


# 这段代码的目的是生成一个形状为 [batch_size, n, seq_len, 1] 的张量 alpha_ent，用于在多头注意力计算中控制注意力分布的形状。
    def get_alpha2(self, x=None, seq_len=70): #x [b,n,d/n]
        alpha_ent = torch.sigmoid(self.multi_alpha_w(x)) + 1  # [b,n,1] # 为什么要将sigmoid激活后的函数+1呢？
        alpha_ent = self.add_value(alpha_ent).unsqueeze(2)  # [b,n,1,1]
        # print("alpha_ent:", alpha_ent)
        alpha_ent = alpha_ent.expand(-1, -1, seq_len, -1)  # [b,n,s,1] expand中-1 是一个特殊的标志，表示保持对应维度的大小不变。
        return alpha_ent # alpha_ent: (b,n,s,1)


    # 意图： 将value中等于 1 的位置修正为 1.00001，但value中似乎没有等于1的值
    def add_value(self, value):
        # if (value == 1).any():  # 检查value(1,2)是否有值为1的元素
        #     print("value contains 1:", value[value == 1])
        mask_value = (value == 1).float()
        value = value.masked_fill(mask_value == 1, 1.00001)
        return value 


    # 将输入张量 x 转换成适合进行多头注意力计算的形状，并返回转置后的张量。
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # new_x_shape:(b,s,num_head,head_size)
        # print("new_x_shape:",new_x_shape)
        x = x.view(*new_x_shape) # *new_x_shape是一种语法，用于解压缩(iterate)可迭代对象new_x_shape
        # print("*new_x_shape:", *new_x_shape)
        return x.permute(0, 2, 1, 3) # x: (b, num_head, s, head_size)


    '''
    这段代码的主要作用是在输入的 q、k、v 上应用了多头自注意力机制，并对处理后的结果进行了残差连接和归一化处理。
    最终返回了 att_v 中最后一个时间步的特征表示 c 和除了最后一个时间步之外的所有时间步的特征表示 x_n
    '''
    def Multi_Self_attention(self, q, k, v, sess_len, mask):
        is_dropout = True
        if is_dropout:
            q_ = self.dropout(self.activate(self.attention_mlp(q)))  # [b,s+1,d] # 经过两层线性层，一层dropout层。
        else:
            q_ = self.activate(self.attention_mlp(q))

        query_layer = self.transpose_for_scores(q_)
        key_layer = self.transpose_for_scores(k)
        value_layer = self.transpose_for_scores(v)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # attention_scores: (b,num_head,s,s)
        # print("attention_scores:", attention_scores.shape)
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        
        alpha_ent = self.get_alpha2(query_layer[:, :, -1, :], seq_len=sess_len) # -1 表示取 s 维度的最后一个元素，因此 s 维度的大小会变成 1。
    # alpha_ent相当于公式中的a_s
        # alpha_ent:(b,num_head,head_size)
        # print("query_layer[:, :, -1, :]:", query_layer[:, :, -1, :].shape, query_layer.shape)
        # print("alpha_ent:", alpha_ent.shape)
        attention_probs = entmax_bisect(attention_scores, alpha_ent, dim=-1)
        #  entmax_bisect根据输入的 attention_scores 和 alpha_ent，在指定的维度 dim=-1 上应用特定的算法（通常是基于二分法的方法），计算每个位置或样本的归一化后的输出。
        context_layer = torch.matmul(attention_probs, value_layer)
    # context_layer相当于公式中的SAtt(Q,K,V)
        # print("content_layer:", context_layer.shape)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous() # context_layer:(b,s,n_head, head_size)
        # .contiguous() 方法用于确保张量在内存中是连续存储的。具体来说，当你需要对一个张量进行操作，例如进行视图变换（如 permute、view）、索引操作或者其他需要连续内存块的操作时，通常需要先调用 .contiguous() 方法，将张量转换为连续的形式，以便后续操作可以正确执行。
        # print("content_layer:", context_layer.shape)
        new_context_layer_shape = context_layer.size()[:-2] + (self.dim,)
        att_v = context_layer.view(*new_context_layer_shape) # att_v: (b,s,d) [512,40,200]
        # print("att_v:", att_v.shape)

        
        # 下面这段代码的目的通常是在自注意力机制中对输入 att_v 进行一系列非线性变换、正则化（dropout），并通过残差连接（residual connection）将处理后的结果与原始输入相结合。
        if is_dropout:
            att_v = self.dropout(self.self_atten_w2(self.activate(self.self_atten_w1(att_v)))) + att_v
        else:
            att_v = self.self_atten_w2(self.activate(self.self_atten_w1(att_v))) + att_v

        att_v = self.LN(att_v)
        c = att_v[:, -1, :].unsqueeze(1)  # [b,d]->[b,1,d]
        x_n = att_v[:, :-1, :]  # [b,s-1,d] # 去除当时传入的0向量，也就是处理后的c
        # att_v[:, :-1, :] 表示对 att_v 在第二个维度上取除了最后一个时间步之外的所有时间步的特征表示。这样得到的 x_n 的形状是 [b, s-1, d]，其中 s 是序列长度，d 是特征维度。
        # print("x_n:", x_n.shape)
        # print("c:", c.shape)
    # c相当于t_n即target of the original graph node， x_n相当于H_t即potential representation of the original graph node
        return c, x_n


    # 经过全局注意力机制的会话表示
    def global_attention(self, target, k, v, mask=None, alpha_ent=1):
        alpha = torch.matmul(torch.relu(k.matmul(self.atten_w1) + target.matmul(self.atten_w2) + self.atten_bias), self.atten_w0.t())
        # print("alpha:", alpha.shape) # [b,s,1]
        if mask is not None: #[b,s]
            mask = mask.unsqueeze(-1)
            alpha = alpha.masked_fill(mask == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=1) #alpha [b,s,1]
        c = torch.matmul(alpha.transpose(1, 2), v)
        # c: [b,1,2d]
        # print("c:", c.shape)
        return c


    # [b,d], [b,d]
    # 返回最终会话表示
    def decoder(self, global_s, target_s):
        if self.is_dropout:
            if self.dataset  == 'Tmall':
                c = self.w_f(torch.cat((global_s, target_s), 2))
            else:
                c = self.dropout(torch.selu(self.w_f(torch.cat((global_s, target_s), 2))))
        else:
            if self.dataset == 'Tmall':
                c = self.w_f(torch.cat((global_s, target_s), 2))
            else:
                c = torch.selu(self.w_f(torch.cat((global_s, target_s), 2)))  # [b,1,4d]
            

        c = c.squeeze() #[b,d]
        # print("torch.cat((global_s, target_s), 2):", torch.cat((global_s, target_s), 2).shape)
        # print("c.shape:", c.shape)
        l_c = (c / torch.norm(c, dim=-1).unsqueeze(1))
        # torch.norm(c, dim=-1) [b] torch.norm(c, dim=-1).unsqueeze(1): [b,d]
        return l_c


    # 计算最终的会话表示、候选物品表示，计算每个物品被推荐的得分
    def compute_scores(self, hidden, mask, target_emb, att_hidden, relation_emb, global_emb):  #Dual_att[b,d], Dual_g[b,d]
        # ht为local_embedding
        ht = hidden[torch.arange(mask.shape[0]).long(), torch.sum(mask, 1) - 1]  # batch_size x latent_size（hidden_size）
        # 使用逗号连接的索引方式，提取每个序列的最后一个非填充物品的隐藏状态。
        # print("torch.sum:", torch.sum(mask,1))
        # print("hidden:", hidden[0])
        q1 = self.linear_one(ht).view(ht.shape[0], 1, ht.shape[1])  # batch_size x 1 x latent_size
        
        q2 = self.linear_two(hidden)  # batch_size x seq_length x latent_size
        # print("q2:", q2.shape)
        sess_global = torch.sigmoid(q1 + q2) #[b,s,d]
        # print("sess_global:", sess_global.shape)
        
        # Sparse Global Attention
        alpha_global = self.get_alpha(x=target_emb, number=1) #[b,1,2d]
        q = target_emb
    
        k = att_hidden #[b,s,2d]
        v = sess_global #[b,s,2d]
    # global_c 相当于 论文中的 H_GAtt
        global_c = self.global_attention(q, k, v, mask=mask, alpha_ent=alpha_global) # [b,1,2d]
        # print("global_c:", global_c.shape)
        sess_final = self.decoder(global_c, target_emb) # sess_final: [b,d]
        
        
        # print("sess_final:", sess_final.shape)
        
        #SIC
        # neighbor_sess = self.FindNeighbor(sess_final + relation_emb)
        # print("shape： ", sess_final.shape, global_emb.shape)
        # neighbor_sess = self.FindNeighbor(sess_final)
        neighbor_sess = self.FindNeighbor(sess_final+relation_emb)
        # neighbor_sess = self.FindNeighbor(sess_final+global_emb+relation_emb)
        # print("sess_final, relation_emb:", sess_final.shape, relation_emb.shape)
        sess_final = sess_final + neighbor_sess

        contrastive_loss = self.ContrastiveLearningModel(sess_final, global_emb)

        b = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)  # b: [n-1,d]
        
        scores = self.w * torch.matmul(sess_final, b.transpose(1, 0))  # [b,d]x[d,n] = [b,n]
        return scores, contrastive_loss


    def forward(self, inputs, A, alias_inputs, A_hat, D_hat, mask, global_inputs, global_items):  # alias_inputs[b,original_repeat_maxsize], A[b,unique_maxsize,2unique_maxsize] inputs[b,unique_maxsize]
        # print("alias_inputs、inputs、A:", alias_inputs.shape, inputs.shape, A.shape)
        seq_emb = self.add_position_embedding(inputs)  # [b,s,2d] # seq_emb中如何融合位置信息？直接将位置求embedding??
        hidden = self.gnn(A, seq_emb)  # (b,s,2d)
        get = lambda i: hidden[i][alias_inputs[i]]
       
        # for i in torch.arange(len(alias_inputs)).long():  
        #     if len(alias_inputs[i][alias_inputs[i] != 0]) >= hidden[i].size(0):
        #         print("alias_inputs, hidden:", len(alias_inputs[i][alias_inputs[i] != 0]) ,hidden[i].shape)
        #         print("alias_inputs[i]:", alias_inputs[i])
        #     if i==511:
        #         print("511!")
        #         break
            
        seq_hidden_gnn = torch.stack([get(i) for i in torch.arange(len(alias_inputs)).long()])  # [b,s,2d] 这里的i为[0，511] seq_hidden_gnn: 该会话索引序列所对应的物品表示

        # zeros = torch.cuda.FloatTensor(seq_hidden_gnn.shape[0], 1, self.dim).fill_(0)  # [b,1,d]
        zeros = torch.zeros(seq_hidden_gnn.shape[0],1,self.dim, dtype=torch.float32, device='cuda') # [b,1,d]
        session_target = torch.cat([seq_hidden_gnn, zeros], 1)  # [b,s+1,d]  # session_target:在每个会话末尾添加一个为0的物品

        sess_len = session_target.shape[1] 
        target_emb, x_n = self.Multi_Self_attention(session_target, session_target, session_target, sess_len, mask)  
        print("seq_hidden_gnn.shape, target_emb.shape:", seq_hidden_gnn.shape, target_emb.shape)
        relation_emb = self.RelationGraph(self.embedding.weight, inputs, A_hat, D_hat, target_emb) # 这里的item_embedding 是否可以修改，全局物品（n_node, hidden）的表示？
        
        global_items = inputs
        global_inputs = global_inputs
        global_alias_inputs = alias_inputs
        
        global_emb = self.graphRecommender(global_items, global_inputs, global_alias_inputs, target_emb)
        # print("global_emb:", global_emb.shape)
        # print("self.embedding.weight:", self.embedding.weight.shape)
        # print("relation_emb:", relation_emb.shape)
        # print("target_emb:", target_emb.shape)
        # print("x_n:", x_n.shape)
        # print("seq_hidden_gnn:", seq_hidden_gnn.shape)
        '''
        如下总结：（不知是否正确）
        seq_hidden_gnn: 批次会话中每个会话的物品序列 （batch_size, original_session_len, 2*hidden_size）
        x_n: 会话中的物品（除最后一个物品）嵌入表示 （batch_size, original_session_len, 2*hidden_size）
        target_emb: 会话中的最后一个物品嵌入表示（batch_size, 1, 2*hidden_size）
        relation_emb: 会话的嵌入表示（batch_size, hidden_size）
        '''
        return seq_hidden_gnn, target_emb, x_n, relation_emb, global_emb


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
    else:
        return variable


def trans_to_cpu(variable):
    if torch.cuda.is_available():
        return variable.cpu()
    else:
        return variable

# 调用模型正向传播，返回标签和预测分数
def forward(model, i, data):
    global_inputs, global_items, alias_inputs, A, items, mask, targets = data.get_slice(i)  # 得到碎片数据：batch中的值
    A_hat, D_hat = data.get_overlap(items) # items:批次会话物品序列[[1,2,3,0,0][1,3,5,5,4],...]
    A_hat = trans_to_cuda(torch.Tensor(A_hat))
    D_hat = trans_to_cuda(torch.Tensor(D_hat))
    global_inputs = trans_to_cuda(torch.Tensor(global_inputs).long())
    global_items = trans_to_cuda(torch.Tensor(global_items).long())
    alias_inputs = trans_to_cuda(torch.Tensor(alias_inputs).long())
    items = trans_to_cuda(torch.Tensor(items).long())
    A = trans_to_cuda(torch.Tensor(np.array(A)).float()) # 批次会话中的入度出度形成的邻接矩阵？？？ try
    mask = trans_to_cuda(torch.Tensor(mask).long())
    hidden, target_emb, att_hidden, relation_emb, global_emb = model(items, A, alias_inputs, A_hat, D_hat, mask, global_inputs, global_items)
    # mask： 掩码，有效长度内为1，否则为0.
    scores, contrastive_loss = model.compute_scores(hidden, mask, target_emb, att_hidden, relation_emb, global_emb)

    return targets, scores, contrastive_loss


# 模型的正向、反向传播；更新参数；返回Hit和MRR评价指标
def train_test(model, train_data, test_data, contrastive_weight):
    print('start training: ', datetime.datetime.now())
    model.train()  
    total_loss = 0.0
    contrastive_weight = contrastive_weight
    print("contrastive_weight:", contrastive_weight)
    slices = train_data.generate_batch(model.batch_size)  
    for i, j in zip(slices, np.arange(len(slices))): 
    # i 会依次取 slices 中的每一个元素（即每个批次的数据），而 j 会依次取 np.arange(len(slices)) 中的每一个元素（即每个批次的编号）。
    # eg: (np.array([0, 1, 2]), 0)
        model.optimizer.zero_grad() 
        # model.optimizer.zero_grad() 是 PyTorch 中的一个常见操作，用于在每次反向传播之前清除优化器中存储的梯度。
        targets, scores, contrastive_loss = forward(model, i, train_data)
        targets = trans_to_cuda(torch.Tensor(targets).long())
        
        loss = model.loss_function(scores, targets - 1)  +  contrastive_weight * contrastive_loss# targets - 1 ?  
        loss.backward()  
        model.optimizer.step()  
        total_loss = total_loss + loss
        if j % int(len(slices) / 5 + 1) == 0:
            print('[%d/%d] Loss: %.4f' % (j, len(slices), loss.item()))
    print('\tLoss:\t%.3f' % total_loss)
    
    # model.scheduler.step() 的作用是根据学习率调度器的策略更新学习率。
    model.scheduler.step()  
    
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
        
    print('start predicting: ', datetime.datetime.now())
    model.eval() 
    '''
    model.eval():评估模式（Evaluation Mode）	是:正向传播	否:反向传播	否：更新参数 Dropout:关闭所有神经元，不再进行随机舍弃 batchNorm:使用在训练阶段计算得到的全局统计数据进行归一化处理。
    '''
    hit, mrr = [], []
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        targets, scores, contrastive_loss = forward(model, i, test_data)
        for K in top_K:
            sub_scores = scores.topk(K)[1]  
            sub_scores = trans_to_cpu(sub_scores).detach().numpy()
            for score, target, mask in zip(sub_scores, targets, test_data.mask):
                if K==20:
                    hit.append(np.isin(target - 1, score))
                metrics['hit%d' % K].append(np.isin(target - 1, score))
                if len(np.where(score == target - 1)[0]) == 0:
                    if K==20:
                        mrr.append(0)
                    metrics['mrr%d' % K].append(0)
                else:
                    if K==20:
                        mrr.append(1 / (np.where(score == target - 1)[0][0] + 1))
                    metrics['mrr%d' % K].append(1 / (np.where(score == target - 1)[0][0] + 1))
    hit = np.mean(hit) * 100
    mrr = np.mean(mrr) * 100
    return hit, mrr, metrics



