#!/usr/bin/env python36
# -*- coding: utf-8 -*-

import argparse
'''
argparse 是 Python 标准库中的一个模块，用于解析命令行参数和选项。
它帮助程序接受命令行输入，从而使得用户可以通过命令行传递参数和选项给 Python 脚本。
'''
import pickle
'''
pickle 是 Python 标准库中的一个模块，用于序列化和反序列化 Python 对象。
序列化是指将 Python 对象转换为字节流的过程，方便存储到文件或通过网络传输；反序列化则是将字节流恢复为原始的 Python 对象。
字节流（byte stream）是指一系列连续的字节（8 位的二进制数据），用于表示数据在计算机内存或在计算机之间的传输。
'''
import time
from utils import Data, split_validation
from model import *
import os
import scipy
from recommender import *
import random

# 这一行代码创建了一个 ArgumentParser 对象，该对象用于处理命令行参数。
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='diginetica', help='dataset name: /Tmall/Nowplaying/diginetica/yoochoose1_4/yoochoose1_64')
parser.add_argument('--batchSize', type=int, default=512, help='input batch size') #64,100,256,512
parser.add_argument('--hiddenSize', type=int, default=100, help='hidden state size')
parser.add_argument('--epoch', type=int, default=30, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # [0.001, 0.0005, 0.0001]
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=5, help='the number of steps after which the learning rate decay 3')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001]
parser.add_argument('--step', type=int, default=1, help='gnn propogation steps')
parser.add_argument('--patience', type=int, default=3, help='the number of epoch to wait before early stop ')
parser.add_argument('--nonhybrid', action='store_true', help='only use the global preference to predict')
parser.add_argument('--validation', action='store_true', help='validation')
# action='store_true'：这表示如果在命令行中提供了 --validation 参数，则将其对应的值设置为 True。如果没有提供该参数，则默认值为 False。
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--w_ne', type=float, default=1.7, help='neighbor weight') #digi：1.7 Tmall 0.9
parser.add_argument('--gama', type=float, default=1.7, help='cos_sim') #digi：1.7

parser.add_argument('--num_attention_heads', type=int, default=5, help='Multi-Att heads')
parser.add_argument('--neighbor_n', type=int, default=3, help='Relation neighbor_n')
parser.add_argument('--contrastive_weight', type=float, default=1.0, help='contrastive_weight')

# 这一行代码调用 parse_args 方法，解析命令行参数并返回一个包含参数值的命名空间对象。
opt = parser.parse_args()
print(opt)

# 设置随机数种子
# random_seed = 2028
# print("random_seed:", random_seed)
# torch.manual_seed(random_seed)
# np.random.seed(random_seed)
# random.seed(random_seed)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(random_seed)

def sparse2sparse(coo_matrix):
    v1 = coo_matrix.data
    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(v1)
    shape = coo_matrix.shape
    sparse_matrix = torch.sparse.LongTensor(i, v, torch.Size(shape))
    return sparse_matrix

def main():
    # pickle.load 函数从打开的文件对象中加载数据。pickle.load 会反序列化文件中的数据，将其恢复为原始的 Python 对象。
    train_data = pickle.load(open('./datasets/' + opt.dataset + '/train.txt', 'rb'))
    if opt.validation:
        train_data, valid_data = split_validation(train_data, opt.valid_portion)
        test_data = valid_data
    else:
        test_data = pickle.load(open('./datasets/' + opt.dataset + '/test.txt', 'rb'))
    
    # dataset:（会话序列，标签）
    train_data_len = len(train_data[0])
    # print("train_data数据格式:", len(train_data[0]), len(train_data[1]))
    # print("train_data第一个数据:", train_data[0][0], train_data[1][0])
    
    train_data = Data(train_data, shuffle=True)
    test_data = Data(test_data, shuffle=False)
    
    # del all_train_seq, g
    if opt.dataset == 'diginetica':
        n_node = 43098
    elif opt.dataset == 'yoochoose1_64' or opt.dataset == 'yoochoose1_4':
        n_node = 37484
    elif opt.dataset == 'Nowplaying':
        n_node = 60417
    elif opt.dataset == 'Tmall':
        n_node = 40728
    elif opt.dataset == 'RetailRocket':
        n_node = 36969
    else:
        n_node = 310

    start = time.time()
    
    global_adj_coo = scipy.sparse.load_npz('datasets/' + opt.dataset + '/adj_global.npz')
    sparse_global_adj = trans_to_cuda(sparse2sparse(global_adj_coo))
    graphRecommender = trans_to_cuda(GraphRecommender(opt, n_node, sparse_global_adj, len_session=train_data.len_max,
                                              n_train_sessions=train_data_len))

    model = trans_to_cuda(SessionGraph(opt, n_node, graphRecommender))
    
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0] # 取得最好的Hit、MRR对应的 epoch 值。
        best_results['metric%d' % K] = [0, 0]

    best_result = [0, 0]
    best_epoch = [0, 0]
    bad_counter = 0
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        hit, mrr, metrics = train_test(model, train_data, test_data, opt.contrastive_weight)
        flag = 0
        
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100  # 将原本的命中率乘以 100，以便将小数形式的百分比转换为百分比形式的整数
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100 
            if best_results['metric%d' % K][0] <= metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
                flag=1
            if best_results['metric%d' % K][1] <= metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                flag=1
            
        for K in top_K:
            print('P@%d: %.4f\tMRR@%d: %.4f\tEpoch: %d,  %d' %
                  (K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))
            
        if hit >= best_result[0]:
            best_result[0] = hit
            best_epoch[0] = epoch
            flag = 1
        if mrr >= best_result[1]:
            best_result[1] = mrr
            best_epoch[1] = epoch
            flag = 1
            
        print('Best Result:')
        print('\tP@20:\t%.4f\tMRR@20:\t%.4f\tEpoch:\t%d,\t%d' % (
        best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
        bad_counter += 1 - flag
        # 早停法。当模型在验证集上连续多次没有取得最佳效果时，提前停止训练。
        if bad_counter >= opt.patience:
            break
    print('-------------------------------------------------------')
    end = time.time()
    print("Run time: %f s" % (end - start)) # 计算的是总体运行时间
    # 逆向 alias_inputs 数组
    # print("逆向 alias_inputs 数组！")
    
    # Save Model
    PATH = "./final_model/" + opt.dataset + "_model.pkl"
    torch.save(model, PATH)


    # # Load the model for testing
    # PATH = "./final_model/" + opt.dataset + "_model.pkl"
    # model = torch.load(PATH)
    # hit, mrr = test(model, test_data)
    #
    # print('Result:')
    # print('\tRecall@20:\t%.4f\tMMR@20:\t%.4f' % (hit, mrr))
    # print('-------------------------------------------------------')
    # end = time.time()
    # print("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()
