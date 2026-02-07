#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python36
# -*- coding: utf-8 -*-
import argparse
import datetime
import os
import pickle
import time
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from datasets.SBRDataSet import SBRDataSet
from model.UIOSBR import LossFunction, UIOSBR
import logging


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",default="diginetica",help="dataset name: diginetica/retailRocket_DSAN/Tmall/Nowplaying")
    parser.add_argument("--device", default="cpu", help="cpu/cuda/mps")
    parser.add_argument("--run_times", default=1, help="experiment run times")
    parser.add_argument("--batchSize", type=int, default=2048, help="input batch size")
    parser.add_argument("--hiddenSize", type=int, default=100, help="hidden state size")
    parser.add_argument("--epoch",type=int,default=15,help="the number of epochs to train for, [digi=12,Tmall=5,Retail=10] ")
    parser.add_argument("--lr", type=float, default=0.003, help="learning rate")  # [0.00128, 0.001, 0.0005, 0.0001]
    parser.add_argument("--lr_dc", type=float, default=0.3, help="learning rate decay rate")
    parser.add_argument("--lr_dc_step",type=int,default=3,help="the number of steps after which the learning rate decay")
    parser.add_argument("--l2", type=float, default=0, help="l2 penalty")  # [0.001, 0.0005, 0.0001, 0.00005, 0.00001] 1e-6
    parser.add_argument("--num_worker", type=int, default=1, help="num worker")

    parser.add_argument("--session_truncated_len", type=int, default=100, help="session truncated length")
    parser.add_argument("--iuignn_step", type=int, default=1, help="IP GNN propogation steps")
    parser.add_argument("--duignn_step", type=int, default=1, help="IO GNN propogation steps")
    parser.add_argument("--patience",type=int,default=10,help="the number of epoch to wait before early stop ")
    parser.add_argument("--gnn_dropout", type=float, default=0.0, help="gnn dropout")
    parser.add_argument("--emb_dropout", type=float, default=0.2, help="emb dropout")
    parser.add_argument("--gru_layer", type=int, default=200, help="gru layer")
    parser.add_argument("--delta", type=float, default=12.5, help="norm factor")  # 12.0 for separate score loss
    parser.add_argument("--use_iuignn", action="store_false", help="whether use IUI-GNN")
    parser.add_argument("--use_duignn", action="store_false", help="whether use DUI-GNN")
    parser.add_argument("--use_seq_item_occur_attn", action="store_false", help="whether use recurrence aware attention")
    parser.add_argument("--use_res", action="store_false", help="whether use review-and-explore-strategy")
    opt = parser.parse_args()
    return opt


def setup_logger( model_save_folder):
    
    level =logging.INFO

    log_name = 'model.log'

    fileHandler = logging.FileHandler(os.path.join(model_save_folder, log_name), mode = 'a')
    fileHandler.setLevel(logging.INFO)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setLevel(logging.INFO)

    formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fileHandler.setFormatter(formatter)
    consoleHandler.setFormatter(formatter)

    logger = logging.getLogger(model_save_folder + log_name)
    logger.setLevel(level)

    logger.addHandler(fileHandler)
    logger.addHandler(consoleHandler)

    return logger


def train(model, train_loader, loss_function, optimizer, scheduler, device):
    model.train()

    total_loss = 0.0
    for data in tqdm(train_loader):
        optimizer.zero_grad()
        alias_item_id_in_session, adj_out, adj_in, unique_item_id_in_session, item_id_in_session, targets, _ = [
            x.to(device) for x in data
        ]
        scores = model.forward(alias_item_id_in_session, adj_out, adj_in, unique_item_id_in_session, item_id_in_session)
        loss = loss_function(scores, targets)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        # scheduler.step()
        total_loss += loss.detach().cpu()
    scheduler.step()
    return total_loss


def test(model, test_loader, device):
    model.eval()

    hit10, mrr10, hit20, mrr20 = [], [], [], []
    with torch.no_grad():
        for data in tqdm(test_loader):
            alias_item_id_in_session, adj_out, adj_in, unique_item_id_in_session, item_id_in_session, targets, _ = [
                x.to(device) for x in data
            ]
            targets = targets.cpu().numpy()
            scores = model.forward(alias_item_id_in_session, adj_out, adj_in, unique_item_id_in_session, item_id_in_session)

            sub_scores = scores.topk(20)[1]
            sub_scores = sub_scores.cpu().detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit20.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr20.append(0)
                else:
                    mrr20.append(1 / (np.where(score == target)[0][0] + 1))
            sub_scores = scores.topk(10)[1]
            sub_scores = sub_scores.cpu().detach().numpy()
            for score, target in zip(sub_scores, targets):
                hit10.append(np.isin(target, score))
                if len(np.where(score == target)[0]) == 0:
                    mrr10.append(0)
                else:
                    mrr10.append(1 / (np.where(score == target)[0][0] + 1))

    hit20 = np.mean(hit20) * 100
    mrr20 = np.mean(mrr20) * 100
    hit10 = np.mean(hit10) * 100
    mrr10 = np.mean(mrr10) * 100

    return hit10, mrr10, hit20, mrr20


def main():
    
    opt = get_parser()

    train_data = pickle.load(open("./datasets/" + opt.dataset + "/train.txt", "rb"))
    test_data = pickle.load(open("./datasets/" + opt.dataset + "/test.txt", "rb"))

    train_data = SBRDataSet(train_data, opt.session_truncated_len)
    test_data = SBRDataSet(test_data, opt.session_truncated_len)

    train_data = torch.utils.data.DataLoader(
        train_data,
        num_workers=opt.num_worker,
        batch_size=opt.batchSize,
        shuffle=True,
        pin_memory=True
    )
    test_data = torch.utils.data.DataLoader(
        test_data,
        num_workers=opt.num_worker,
        batch_size=opt.batchSize,
        shuffle=False,
        pin_memory=True,
    )


    for _ in range(opt.run_times):
        
        model_save_folder = './' + opt.dataset +'-save-' + datetime.datetime.now().strftime( '%Y%m%d_%H%M%S_%f') + '/'
        if os.path.exists(model_save_folder) == False:
            os.makedirs(model_save_folder)
            logger = setup_logger(model_save_folder)
        
        logger.info("hyper parameters")
        logger.info(str(opt))

        device = torch.device(opt.device)
        
        model = UIOSBR(opt)

        model = model.to(device)

        loss_function = LossFunction()

        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.l2)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)

        start = time.time()
        best_result = [0, 0, 0, 0]
        best_epoch = [0, 0, 0, 0]
        patience_counter = 0

        for epoch in range(opt.epoch):
            logger.info("epoch [%d/%d]" % (epoch+1, opt.epoch))

            train_loss = train(model, train_data, loss_function, optimizer, scheduler, device)
            
            logger.info("Loss:{}".format(train_loss))
            
            logger.info("start predicting")
            
            hit10, mrr10, hit20, mrr20 = test(model, test_data, device)
            
            patience_counter += 1
            if hit20 >= best_result[0]:
                best_result[0] = hit20
                best_epoch[0] = epoch
                patience_counter = 0
            if mrr20 >= best_result[1]:
                best_result[1] = mrr20
                best_epoch[1] = epoch
                patience_counter = 0
            if hit10 >= best_result[2]:
                best_result[2] = hit10
                best_epoch[2] = epoch
                patience_counter = 0
            if mrr10 >= best_result[3]:
                best_result[3] = mrr10
                best_epoch[3] = epoch
                patience_counter = 0

            logger.info("Best Result:")
            logger.info("\t P@20:\t {} \t MRR@20:\t {} \tEpoch:\t {},\t {}".format(best_result[0], best_result[1], best_epoch[0], best_epoch[1]))
            logger.info("\t P@10:\t {} \t MRR@10:\t {} \tEpoch:\t {},\t {}".format(best_result[2], best_result[3], best_epoch[2], best_epoch[3]))
            logger.info("Current Result:")
            logger.info("\t P@20:\t {} \t MRR@20:\t {}".format(hit20, mrr20))
            logger.info("\t P@10:\t {} \t MRR@10:\t {}".format(hit10, mrr10))

            torch.save(model.state_dict(), model_save_folder + "last_model.pth")
            
            if patience_counter >= opt.patience:
                break
        end = time.time()
        logger.info("Run time: %f s" % (end - start))


if __name__ == '__main__':
    main()

