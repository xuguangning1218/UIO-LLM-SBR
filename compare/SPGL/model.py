import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module
import torch.nn.functional as F
import torch.sparse
from scipy.sparse import coo_matrix
import time
import random
from numba import jit
import heapq

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


class ItemConv(Module):
    def __init__(self, layers, emb_size=100):
        super(ItemConv, self).__init__()
        self.emb_size = emb_size
        self.layers = layers
         
        # w_item: ModuleList，包含layers数量的线性层。每个线性层大小为（emb_size, emb_size）且没有偏置项
        self.w_item = nn.ModuleList([nn.Linear(self.emb_size, self.emb_size, bias=False) for i in range(self.layers)])
        # attention: 线性层，计算注意力得分，从emb_size映射到1
        self.attention = nn.Linear(self.emb_size, 1)
        
    def forward(self, adjacency, embedding):
        # 度矩阵 
        degree = adjacency.sum(axis=0).reshape(1, -1)

        degree = 1/degree
        adjacency = adjacency.multiply(degree) # A/D
        
        values = adjacency.data
        # 获取邻接矩阵的行索引、列索引，将其垂直堆叠在一起
        indices = np.vstack((adjacency.row, adjacency.col))
        i = torch.LongTensor(indices)
        v = torch.FloatTensor(values)
        shape = adjacency.shape
        # adjacency = torch.sparse.FloatTensor(i, v, torch.Size(shape))
        adjacency = torch.sparse_coo_tensor(i, v, torch.Size(shape))
        
        item_embeddings = embedding
        item_embedding_layer0 = item_embeddings
        final = item_embedding_layer0
        for i in range(self.layers):
            # Attention mechanism
            attention_scores = self.attention(item_embeddings)
            attention_weights = F.softmax(attention_scores, dim=0)
            
            item_embeddings = trans_to_cuda(self.w_item[i])(item_embeddings)
            item_embeddings = torch.sparse.mm(trans_to_cuda(adjacency), item_embeddings*attention_weights)
           
            item_embeddings = F.normalize(item_embeddings, dim=-1, p=2)
            final = torch.add(item_embeddings, final)
            
        item_embeddings = final/(self.layers+1)
        # item_embeddings:(n_nodes, embedding_size)
        return item_embeddings
    
class COTREC(Module):
    def __init__(self, adjacency, n_node, lr, layers, l2, dataset, emb_size=100, batch_size=100,
                 temperature=0.1, 
                 item_cl_loss_weight=100,
                 sampled_item_size=30000,
                 top_k=10):
        super(COTREC, self).__init__()
        self.emb_size = emb_size
        self.batch_size = batch_size
        self.n_node = n_node
        self.dataset = dataset
        self.L2 = l2
        self.lr = lr
        self.layers = layers
    
        self.K = 10
        self.w_k = 10
        self.num = 5000
        self.adjacency = adjacency
        self.embedding = nn.Embedding(self.n_node, self.emb_size)
        self.pos_len = 200
        if self.dataset == 'retailRocket_DSAN':
            self.pos_len = 300
        self.pos_embedding = nn.Embedding(self.pos_len, self.emb_size)
        self.ItemGraph = ItemConv(self.layers)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.emb_size, self.emb_size))
        self.w_2 = nn.Parameter(torch.Tensor(self.emb_size, 1))
        self.w_i = nn.Linear(self.emb_size, self.emb_size)
        self.w_s = nn.Linear(self.emb_size, self.emb_size)
        self.glu1 = nn.Linear(self.emb_size, self.emb_size)
        self.glu2 = nn.Linear(self.emb_size, self.emb_size, bias=False)

        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.init_parameters()
        
        # : compute contrastive loss among item embeddings
        self.temperature = temperature
        self.item_cl_loss_weight = item_cl_loss_weight
        self.sampled_item_size = sampled_item_size
        self.top_k = top_k
        
    def init_parameters(self):
        stdv = 1.0 / math.sqrt(self.emb_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    # 生成会话嵌入表示
    def generate_sess_emb(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = trans_to_cuda(torch.FloatTensor(1, self.emb_size).fill_(0))
        
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = trans_to_cuda(torch.FloatTensor(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size).fill_(0))
        
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)

         
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        pos_emb = self.pos_embedding.weight[:len]
        pos_emb = pos_emb.unsqueeze(0).repeat(self.batch_size, 1, 1)

        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.matmul(torch.cat([pos_emb, seq_h], -1), self.w_1)
        nh = torch.tanh(nh)
        nh = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        return select

    def generate_sess_emb_npos(self, item_embedding, session_item, session_len, reversed_sess_item, mask):
        zeros = trans_to_cuda(torch.zeros(1, self.emb_size, dtype=torch.float))
        # zeros = torch.zeros(1, self.emb_size)
        item_embedding = torch.cat([zeros, item_embedding], 0)
        get = lambda i: item_embedding[reversed_sess_item[i]]
        seq_h = trans_to_cuda(torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size, dtype=torch.float))
        # seq_h = torch.zeros(self.batch_size, list(reversed_sess_item.shape)[1], self.emb_size)
        
        for i in torch.arange(session_item.shape[0]):
            seq_h[i] = get(i)
            
        
        hs = torch.div(torch.sum(seq_h, 1), session_len)
        mask = mask.float().unsqueeze(-1)
        len = seq_h.shape[1]
        hs = hs.unsqueeze(-2).repeat(1, len, 1)
        nh = torch.sigmoid(self.glu1(seq_h) + self.glu2(hs))
        beta = torch.matmul(nh, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * seq_h, 1)
        
        '''
            nh: torch.Size([100, 35, 100])
            beta: torch.Size([100, 35, 1])
            select: torch.Size([100, 100])
        '''
        return select

    def forward(self, session_item, session_len, reversed_sess_item, mask, epoch, tar, train, diff_mask):
        if train:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                # for Tmall dataset, we do not use position embedding to learn temporal order
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len,reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            
            
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            
        else:
            item_embeddings_i = self.ItemGraph(self.adjacency, self.embedding.weight)
            if self.dataset == 'Tmall':
                sess_emb_i = self.generate_sess_emb_npos(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            else:
                sess_emb_i = self.generate_sess_emb(item_embeddings_i, session_item, session_len, reversed_sess_item, mask)
            sess_emb_i = self.w_k * F.normalize(sess_emb_i, dim=-1, p=2)
            item_embeddings_i = F.normalize(item_embeddings_i, dim=-1, p=2)
            
            
            scores_item = torch.mm(sess_emb_i, torch.transpose(item_embeddings_i, 1, 0))
            loss_item = self.loss_function(scores_item, tar)
            
            
        cl_loss = 0
        bs, _ = item_embeddings_i.shape # (num_nodes, bsz_size)
        logits = torch.div(
            torch.matmul(item_embeddings_i, item_embeddings_i.T),
            self.temperature)
        # torch.topk：从张量logits中沿指定维度dim=-1(即最后一个维度)选择前k个最大的元素，并返回这些元素及其索引
        # topk_logits、topk_index:（n_nodes, top_k）
        topk_logits, topk_index = torch.topk(logits, k=self.top_k, dim=-1) # topk_logits: 最大值 topk_index:最大索引
        if torch.cuda.is_available():
            cl_item_loss = self.item_cl_loss_weight * self.loss_function(topk_logits, torch.zeros(bs, dtype=torch.int64).cuda())
            # 假定标签 传入交叉熵损失函数
        else:
            cl_item_loss = self.item_cl_loss_weight * self.loss_function(topk_logits, torch.zeros(bs, dtype=torch.int64))
        cl_loss += cl_item_loss
            
        
        return  loss_item, scores_item, cl_loss

# 调用COTREC模型的forward函数
def forward(model, i, data, epoch, train):
    tar, session_len, session_item, reversed_sess_item, mask, diff_mask = data.get_slice(i)
    diff_mask = trans_to_cuda(torch.Tensor(diff_mask).long())
    
    session_item = trans_to_cuda(torch.Tensor(session_item).long())
    session_len = trans_to_cuda(torch.Tensor(session_len).long())
    
    tar = trans_to_cuda(torch.Tensor(tar).long())
    mask = trans_to_cuda(torch.Tensor(mask).long())
    reversed_sess_item = trans_to_cuda(torch.Tensor(reversed_sess_item).long())
    loss_item, scores_item, cl_item_loss = model(session_item, session_len, reversed_sess_item, mask, epoch,tar, train, diff_mask)
    return tar, scores_item,  loss_item,  cl_item_loss


@jit(nopython=True)
# 找出前K个最大得分的元素，并返回他们的索引
def find_k_largest(K, candidates):
    n_candidates = []
    for iid, score in enumerate(candidates[:K]):
        n_candidates.append((score, iid))
    heapq.heapify(n_candidates) # 将n_candidates转换为最小堆
    # 列表中第K个元素及其后的元素，如果大于当前最小得分的候选元素，则替换该元素，并重新维护堆的结构
    for iid, score in enumerate(candidates[K:]):
        if score > n_candidates[0][0]:
            heapq.heapreplace(n_candidates, (score, iid + K))
    # n_candidates按照得分从大到小进行排序
    n_candidates.sort(key=lambda d: d[0], reverse=True)
    ids = [item[1] for item in n_candidates]
    # k_largest_scores = [item[0] for item in n_candidates]
    return ids#, k_largest_scores


def train_test(model, train_data, test_data, epoch):
    print('start training: ', datetime.datetime.now())
    total_loss = 0.0
    slices = train_data.generate_batch(model.batch_size)
    for i in slices:
        model.zero_grad()
        tar, scores_item, loss_item, cl_item_loss = forward(model, i, train_data, epoch, train=True)
        loss = loss_item + cl_item_loss
        loss.backward()
        model.optimizer.step()
        total_loss += loss.item()
    print('\tLoss:\t%.3f' % total_loss)
    top_K = [5, 10, 20]
    metrics = {}
    for K in top_K:
        metrics['hit%d' % K] = []
        metrics['mrr%d' % K] = []
    print('start predicting: ', datetime.datetime.now())

    model.eval()
    slices = test_data.generate_batch(model.batch_size)
    for i in slices:
        tar,scores_item, loss_item, cl_item_loss  = forward(model, i, test_data, epoch, train=False)
        scores = trans_to_cpu(scores_item).detach().numpy()
        
        index = []
        for idd in range(model.batch_size):
            index.append(find_k_largest(20, scores[idd]))
        index = np.array(index)
        tar = trans_to_cpu(tar).detach().numpy()
        
        for K in top_K:
            # (prediction, target)的元组：index的每一行（前k列）和 tar 中的元素相应配对
            for prediction, target in zip(index[:, :K], tar):
                metrics['hit%d' % K].append(np.isin(target, prediction))
                # np.where(prediction == target)[0] 中的 [0] 是用于从返回的元组中提取第一个元素。eg:(array([2]),)
                if len(np.where(prediction == target)[0]) == 0:
                    metrics['mrr%d' % K].append(0)
                else:
                    # np.where(prediction == target)[0][0]：首先 [0] 提取索引数组，然后 [0] 再次提取这个数组的第一个元素。
                    metrics['mrr%d' % K].append(1 / (np.where(prediction == target)[0][0] + 1))
    return metrics, total_loss


