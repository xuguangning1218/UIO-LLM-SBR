#!/usr/bin/env python36
# -*- coding: utf-8 -*-
"""
Created on July, 2018

@author: Tangrizzly
"""

import networkx as nx
import numpy as np

#us_pois:[1,2,3,0,0,0] us_masks:[1,1,1,0,0,0] 返回一定长度的会话物品序列、对应掩码、真实会话物品长度
def data_masks(all_usr_pois, item_tail):
    us_lens = [len(upois) for upois in all_usr_pois]
    len_max = max(us_lens)
    # print("len_max指的是所有数据集中最长的会话:", len_max) # 训练集和测试集
    us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
    us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
    return us_pois, us_msks, len_max

# 划分数据集元组为训练集和验证集
def split_validation(train_set, valid_portion):
    train_set_x, train_set_y = train_set
    n_samples = len(train_set_x)
    sidx = np.arange(n_samples, dtype='int32')
    np.random.shuffle(sidx)
    n_train = int(np.round(n_samples * (1. - valid_portion)))
    valid_set_x = [train_set_x[s] for s in sidx[n_train:]]
    valid_set_y = [train_set_y[s] for s in sidx[n_train:]]
    train_set_x = [train_set_x[s] for s in sidx[:n_train]]
    train_set_y = [train_set_y[s] for s in sidx[:n_train]]

    return (train_set_x, train_set_y), (valid_set_x, valid_set_y)


class Data():
    def __init__(self, data, shuffle=False, graph=None):
        inputs = data[0]
        inputs, mask, len_max = data_masks(inputs, [0])
        self.inputs = np.asarray(inputs) # numpy.asarray()函数的基本功能是将输入转换为NumPy数组。
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        self.length = len(inputs)
        self.shuffle = shuffle
        self.graph = graph


# matrix：对称矩阵，对角线为1，会话间相似度交并比。度矩阵的倒数
    def get_overlap(self, sessions):
        matrix = np.zeros((len(sessions), len(sessions)))
        for i in range(len(sessions)):
            seq_a = set(sessions[i])
            seq_a.discard(0)
            for j in range(i+1, len(sessions)):
                seq_b = set(sessions[j])
                seq_b.discard(0)
                overlap = seq_a.intersection(seq_b)
                ab_set = seq_a | seq_b
                matrix[i][j] = float(len(overlap))/float(len(ab_set))
                matrix[j][i] = matrix[i][j]
        matrix = matrix + np.diag([1.0]*len(sessions))
        degree = np.sum(np.array(matrix), 1)
        # 对矩阵的每一行求和，得到矩阵的每个会话和其他会话的相似度总和
        degree = np.diag(1.0/degree)
        # print("len(session):", len(sessions)) 512
        # 生成度数的倒数的对角矩阵
        '''
        [
            [0.625, 0, 0],
            [0, 0.588, 0],
            [0, 0, 0.526]
        ]
        '''
        return matrix, degree

# 将数据集中的数据划分为若干批次，同时由于最后一个批次可能不满 batch_size，需要将其长度调整为实际剩余的数据量。
    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

# 生成物品序列对应的索引、邻接矩阵、索引序列所对应的实际物品、掩码和标签值
    def get_slice(self, i):
        inputs, mask, targets = self.inputs[i], self.mask[i], self.targets[i]
        
        global_inputs = self.inputs[i]
        global_items = []
        
        items, n_node, A, alias_inputs = [], [], [], []
        for u_input in inputs:
            n_node.append(len(np.unique(u_input))) # n_node: 记录每个会话中唯一物品数量
        max_n_node = np.max(n_node)
        for u_input in inputs:
            
            node = np.unique(u_input) # 每个会话中的唯一物品
            items.append(node.tolist() + (max_n_node - len(node)) * [0])
            global_items.append(node.tolist() + (self.len_max - len(node)) * [0])
            global_alias_inputs = [np.where(node == i)[0][0] for i in u_input]
            u_A = np.zeros((max_n_node, max_n_node))
            
            node_dict = {n:0 for n in node}
            # print("u_input:", u_input)
            for i in np.arange(len(u_input) - 1):
                if u_input[i + 1] == 0:
                    break
                # u_input:[1,2,3,3,4] node:[1,2,3,4]返回的是索引相等 (array([0]),) [0] 0 这里的array([0])其中0指的是node中相等位置的下标 
                # node == u_input[i]：这会生成一个布尔数组，数组的每个元素表示 node 中对应位置的元素是否等于 u_input[i]。例如，如果 node = [1, 2, 3, 4] 且 u_input[i] = 3，那么 node == u_input[i] 会返回 [False, False, True, False]。
                u = np.where(node == u_input[i])[0][0]
                v = np.where(node == u_input[i + 1])[0][0]
                # u_A[u][v] = 1
                if v not in node_dict:
                    node_dict[v] = 0
                node_dict[v]+=1
                u_A[u][v] += node_dict[v]
            # print("u_A[u][v]:", u_A)
            # u_A 记录的是若两物品相邻，则（会话中的唯一物品数最多，会话中的唯一物品数最多）矩阵的元素升序，下标为1。
            u_sum_in = np.sum(u_A, 0) # np.sum(u_A, 0) 的作用是对数组 u_A 的列（把同一列每行的值相加）进行求和。 当前会话中每个物品的入度
            u_sum_in[np.where(u_sum_in == 0)] = 1 # 入度为 0 的物品替换为 1
            u_A_in = np.divide(u_A, u_sum_in) # 归一化矩阵的列
            u_sum_out = np.sum(u_A, 1)
            u_sum_out[np.where(u_sum_out == 0)] = 1
            u_A_out = np.divide(u_A.transpose(), u_sum_out) 
            # u_sum_out 是一个一维数组，其长度等于 u_A.transpose() 的行数（即原矩阵的列数），所以它会被广播到每一列上进行除法操作。
            u_A = np.concatenate([u_A_in, u_A_out]).transpose()
            # 将 u_A_in 和 u_A_out 沿着指定的轴（默认是轴 0，即按行拼接）合并成一个新的矩阵。
            A.append(u_A) # u_A（2max_n_node, max_n_node）
            alias_inputs.append([np.where(node == i)[0][0] for i in u_input])
            
            # 逆向 alias_inputs 数组
            # for node in alias_inputs:
            #     node.reverse()
            
            # 将每个会话的物品 u_input 中的物品映射到它们在 node 中的索引，并将这些索引列表添加到 alias_inputs 列表中
        # print("max_n_node:", max_n_node)
        # print("utils中构建alias_inputs、items、A的结果：", len(alias_inputs[1]), len(items[1]), len(A[1]))
        return global_inputs, global_items, alias_inputs, A, items, mask, targets
        '''
        alias_inputs：每个会话中的物品序列，映射到在 node 中的索引位置。
        A：每个会话的邻接矩阵，包含入度和出度的归一化信息。
        items：每个会话的物品列表，用0填充到 max_n_node 的长度。
        mask：原始的掩码（未变）。
        targets：原始的标签（未变）。
        alias_inputs 对应的是这些物品在 node 中的位置索引。items 对应的是物品的实际值。
        '''
        
