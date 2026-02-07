import time
from util import Data
from model import train_test, COTREC, trans_to_cuda
import torch
import random
import numpy as np
import os
import argparse
import pickle
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='retailrocket', help='dataset name: retailrocket/diginetica/Nowplaying/sample')
parser.add_argument('--epoch', type=int, default=30, help='number of epochs to train for')
parser.add_argument('--batchSize', type=int, default=100, help='input batch size')
parser.add_argument('--embSize', type=int, default=100, help='embedding size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--layer', type=int, default=5, help='the number of layer used')

parser.add_argument('--temperature', type=float, default=0.1)
parser.add_argument('--item_cl_loss_weight', type=float, default=1)
parser.add_argument('--sampled_item_size', type=int, default=35000)
parser.add_argument('--top_k', type=int, default=50)
parser.add_argument('--hoplimit', type=int, default=3)

parser.add_argument('--seed', type=int, default=42, help='random seed')  # 可以更换为其他数字

opt = parser.parse_args()
print(opt) 

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def main():
    set_random_seed(opt.seed)
    
    train_data = pickle.load(open('/home/xuguangning/work/UIO-LLM-SBR/datasets/' + opt.dataset + '/train.txt', 'rb'))
    test_data = pickle.load(open('/home/xuguangning/work/UIO-LLM-SBR/datasets/' + opt.dataset + '/test.txt', 'rb'))
    all_train = pickle.load(open('/home/xuguangning/work/UIO-LLM-SBR/datasets/' + opt.dataset + '/all_train_seq.txt', 'rb'))

    n_node = {
        "diginetica": 43097,
        "Tmall": 40727,
        "Nowplaying": 60416,
        "retailRocket_DSAN": 36968
    }
    
    train_data = Data(train_data,all_train,opt.hoplimit, shuffle=True, n_node=n_node[opt.dataset])
    test_data = Data(test_data,all_train, opt.hoplimit, shuffle=True, n_node=n_node[opt.dataset])
    model = trans_to_cuda(COTREC(adjacency=train_data.adjacency,n_node=n_node[opt.dataset],lr=opt.lr, l2=opt.l2,layers=opt.layer,emb_size=opt.embSize, batch_size=opt.batchSize,dataset=opt.dataset,
                    temperature=opt.temperature,
                    item_cl_loss_weight=opt.item_cl_loss_weight,
                    sampled_item_size=opt.sampled_item_size,
                    top_k=opt.top_k
                    ))
    top_K = [5, 10, 20]
    best_results = {}
    for K in top_K:
        best_results['epoch%d' % K] = [0, 0]
        best_results['metric%d' % K] = [0, 0]
        
    for epoch in range(opt.epoch):
        print('-------------------------------------------------------')
        print('epoch: ', epoch)
        
        metrics, total_loss = train_test(model, train_data, test_data, epoch)
        for K in top_K:
            metrics['hit%d' % K] = np.mean(metrics['hit%d' % K]) * 100
            metrics['mrr%d' % K] = np.mean(metrics['mrr%d' % K]) * 100
            if best_results['metric%d' % K][0] < metrics['hit%d' % K]:
                best_results['metric%d' % K][0] = metrics['hit%d' % K]
                best_results['epoch%d' % K][0] = epoch
            if best_results['metric%d' % K][1] < metrics['mrr%d' % K]:
                best_results['metric%d' % K][1] = metrics['mrr%d' % K]
                best_results['epoch%d' % K][1] = epoch
                    
        print(metrics)
        for K in top_K:
            print('train_loss:\t%.4f\tRecall@%d: %.4f\tMRR%d: %.4f\tEpoch: %d,  %d' %
                  (total_loss, K, best_results['metric%d' % K][0], K, best_results['metric%d' % K][1],
                   best_results['epoch%d' % K][0], best_results['epoch%d' % K][1]))


if __name__ == '__main__':
    main()