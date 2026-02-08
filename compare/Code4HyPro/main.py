import os
import datetime
import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from model import HyPro, trans_to_cuda, forward
from utils import proto_to_items, renumberItems, split_validation, Data, get_recall, get_mrr, get_hr
from tqdm import tqdm

torch.autograd.set_detect_anomaly(True)
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False

# os.environ['CUDA_VISIBLE_DEVICES'] = '5'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(3)

parser = argparse.ArgumentParser()
# Datasets
parser.add_argument('--path', default='/home/xuguangning/work/UIO-LLM-SBR/datasets/', help='path of datasets')
parser.add_argument('--dataset', default='Tmall', help='Tmall-2/diginetica-2/retailrocket-2')
# Training
parser.add_argument('--epoch', type=int, default=15, help='number of epochs')
parser.add_argument('--batchSize', type=int, default=100, help='batch size')
parser.add_argument('--l2', type=float, default=1e-5, help='l2 penalty')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--lr_dc', type=int, default=0.6, help='learning rate decay.')
parser.add_argument('--lr_dc_step', type=int, default=1, help='the number of steps after which the learning rate decay.')
# Model
parser.add_argument('--embedding', type=int, default=100, help='embedding size of items')
parser.add_argument('--posembedding', type=int, default=100, help='embedding size of position embedding')
parser.add_argument('--gnn_layer', type=float, default=1, help='model depth')
parser.add_argument('--en_layer', type=float, default=1, help='global-local enhance layer')
parser.add_argument('--heads', type=float, default=1, help='number of heads')
parser.add_argument('--k', type=float, default=500, help='number of cluster centroids')
parser.add_argument('--threshold', type=float, default=0.0, help='standard of decoupling')
parser.add_argument('--p', type=float, default=0.99, help='probability of item masking')
parser.add_argument('--theta', type=float, default=1.0, help='scale of cross-scale contrastive learning')
parser.add_argument('--temp', type=float, default=0.1, help='temperature for contrastive learning')
parser.add_argument('--semantic_cluster', default=False, help='with or without semantic clustering')
parser.add_argument('--dropout_in', type=float, default=0.1, help='Dropout rate at input layer.')
parser.add_argument('--dropout_hid', type=float, default=0.1, help='Dropout rate at hidden layer.')
parser.add_argument('--isvalidation', action='store_true', help='validation')
parser.add_argument('--seed', type=int, default=42, help='random seed')

opt = parser.parse_args()
print(opt)
random_seed = opt.seed
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
if torch.cuda.is_available():
    print('model is on gpu')
else:
    print('model is on cpu')

# -------------------- Data Loading --------------------- #
train = pickle.load(open(opt.path + opt.dataset + '/train.txt', 'rb'))
test = pickle.load(open(opt.path + opt.dataset + '/test.txt', 'rb'))
item_topo_proto = pickle.load(open(opt.path + opt.dataset + '/topo_protos_ratio_500' + '.pkl', 'rb'))  # [1:]
item_to_topo = torch.tensor(item_topo_proto)

# ----------------- Data Preprocessing ------------------ #
topo_to_items = proto_to_items(item_to_topo)

train_x = train[0]
train_y = train[1]
test_x = test[0]
test_y = test[1]

train_x, train_y, test_x, test_y, item_set, item_dict = renumberItems(train_x, train_y, test_x, test_y)
all_items = torch.Tensor([i for i in item_set]).long()
num_node = len(all_items) + 1
print('the number of nodes is :{}'.format(num_node))
print('the number of clusters is is :{}'.format(opt.k))

train = (train_x, train_y)
test = (test_x, test_y)

if opt.isvalidation:
    print('start validation')
    train_data, valid_data = split_validation(train, 0.2)
    test_data = valid_data
else:
    train_data = train
    test_data = test

length = len(max(train_x, key=len))
train_data = Data(train_data, length)
test_data = Data(test_data, length)

# ----------------- Model Definition ------------------ #
model = trans_to_cuda(HyPro(opt, item_to_topo, topo_to_items, num_node, length))

# ------------------ Train and Test ------------------- #
print('--------------------------Start Training--------------------------')
train_loader = torch.utils.data.DataLoader(train_data, num_workers=4, batch_size=model.batch_size,
                                           shuffle=True, pin_memory=True)
opti = model.optimizer
criterion = nn.CrossEntropyLoss().cuda()
best_result = [0, 0]
best_epoch = [0, 0]
best_result = 0
best_model_result = []
for epoch in range(opt.epoch):
    model.train()
    session_reps = []
    print('---------------This is epoch: {}---------------'.format(epoch))
    for step, data in enumerate(train_loader):
        target = data[0]
        if opt.semantic_cluster == True and step % 5 == 0:
            model.e_step()
        Result, cl = forward(model, data, step)
        NP_Loss = trans_to_cuda(criterion(Result, trans_to_cuda(target - 1).long()))
        Loss = NP_Loss + cl
        if step % 1000 == 0:
            print("Loss:{}".format(Loss))
            print("CL_Loss:{}".format(cl))

        opti.zero_grad()
        Loss.backward()
        opti.step()

    model.scheduler.step()

    print('-------------------Start Predicting---------------------: ', datetime.datetime.now())

    model.eval()
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=model.batch_size,
                                              shuffle=False, pin_memory=True)
    y_pre_1_all = torch.LongTensor().cuda()
    y_pre_1_all_10 = torch.LongTensor()
    y_pre_1_all_5 = torch.LongTensor()

    predict_nums = [i + 1 for i in range(20)]
    hit = [[] for _ in range(len(predict_nums))]
    mrr = [[] for _ in range(len(predict_nums))]
    ndcg = [[] for _ in range(len(predict_nums))]

    for data in test_loader:
        with torch.no_grad():
            max_len_test = test_data.get_max_len()
            y_pre_1, output = model.predict(data, 20)

            y_pre_1_all = torch.cat((y_pre_1_all, y_pre_1), 0)
            y_pre_1_all_10 = torch.cat((y_pre_1_all_10, y_pre_1.cpu()[:, :10]), 0)
            y_pre_1_all_5 = torch.cat((y_pre_1_all_5, y_pre_1.cpu()[:, :5]), 0)
    # Graph
    recall = get_recall(y_pre_1_all, trans_to_cuda(torch.Tensor(test_y)).long().unsqueeze(1) - 1)
    recall_10 = get_recall(y_pre_1_all_10, torch.Tensor(test_y).unsqueeze(1) - 1)
    recall_5 = get_recall(y_pre_1_all_5, torch.Tensor(test_y).unsqueeze(1) - 1)
    new_test_y = [x - 1 for x in test_y]
    hit = get_hr(y_pre_1_all, new_test_y)
    hit_10 = get_hr(y_pre_1_all_10, new_test_y)
    hit_5 = get_hr(y_pre_1_all_5, new_test_y)
    mrr = get_mrr(y_pre_1_all, trans_to_cuda(torch.Tensor(test_y)).long().unsqueeze(1) - 1)
    mrr_10 = get_mrr(y_pre_1_all_10, torch.Tensor(test_y).unsqueeze(1) - 1)
    mrr_5 = get_mrr(y_pre_1_all_5, torch.Tensor(test_y).unsqueeze(1) - 1)

    if best_result < recall:
        best_result = recall
        best_model_result = [recall_5, recall_10, recall, mrr_5, mrr_10, mrr]
        state_dict = {"model": model.state_dict(), "embedding": model.embedding.state_dict()}
        # torch.save(state_dict, 'saves/' + 'dg_HyPro_wo_in.pth')

    print("Results of LocalSession:\n")
    print("Recall@20: " + "%.4f" % recall + " Recall@10: " + "%.4f" % recall_10 + "  Recall@5:" + "%.4f" % recall_5)
    print("HiT@20: " + "%.4f" % hit + " HiT@10: " + "%.4f" % hit_10 + "  HiT@5:" + "%.4f" % hit_5)
    print("MRR@20:" + "%.4f" % mrr.tolist() + " MRR@10:" + "%.4f" % mrr_10.tolist() + " MRR@5:" + "%.4f" % mrr_5.tolist())
    print("\n")
    print("Best Results:\n")
    print(best_model_result)

    torch.cuda.empty_cache()
