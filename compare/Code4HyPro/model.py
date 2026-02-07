import faiss
import torch
import math
import numpy as np
from torch import nn
from torch.nn import Module
from entmax import entmax_bisect
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature))
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float())

    def forward(self, output1, target):
        SIZE = output1.shape[0]

        target_expanded = target.unsqueeze(1) 
        mask = (target_expanded == target)
        penalty = self.calculate_penalty(output1, mask)
        output2 = (mask.float() @ output1)  


        output1 = F.normalize(output1, dim=1)
        output2 = F.normalize(output2, dim=1)

        representations = torch.cat([output1, output2], dim=0)
        similarity_matrix = torch.mm(representations, representations.t().contiguous())
        sim_ij = torch.diag(similarity_matrix, SIZE)
        sim_ji = torch.diag(similarity_matrix, -SIZE)
        positives = torch.cat([sim_ij, sim_ji], dim=0)
        nominator = torch.exp(positives / self.temperature)
        negatives_mask = self.sample_mask(target)
        denominator = negatives_mask * torch.exp(similarity_matrix / self.temperature)
        loss_partial = -torch.log(nominator / (torch.sum(denominator, dim=1) * penalty + 1e-7))
        loss = torch.sum(loss_partial) / (2 * SIZE)
        return loss

    def calculate_penalty(self, sessions, mask):
        neighbor_sessions = torch.matmul(mask.float(), sessions)

        valid_count = mask.sum(dim=1, keepdim=True)  
        mean = neighbor_sessions.sum(dim=1, keepdim=True) / valid_count  
        squared_diff = ((neighbor_sessions - mean) ** 2)
        variance = squared_diff.sum(dim=1) / valid_count.squeeze(-1)  
        penalty = 1 + 0.1 * self.normalize_matrix(variance)
        penalty = torch.cat([penalty, penalty], dim=0)
        return penalty

    def normalize_matrix(self, matrix):
        min_vals, _ = torch.min(matrix, dim=0, keepdim=True)
        max_vals, _ = torch.max(matrix, dim=0, keepdim=True)
        normalized_matrix = (matrix - min_vals) / (max_vals - min_vals + 1e-7)  
        return normalized_matrix

    def sample_mask(self, targets):
        targets = targets.cpu().numpy()
        targets = np.concatenate([targets, targets])

        cl_dict = {}
        for i, target in enumerate(targets):
            cl_dict.setdefault(target, []).append(i)
        mask = np.ones((len(targets), len(targets)))
        for i, target in enumerate(targets):
            for j in cl_dict[target]:
                if abs(j - i) != len(targets) / 2:  
                    mask[i][j] = 0
        return trans_to_cuda(torch.Tensor(mask)).float()



class LocalAggregator(nn.Module):
    def __init__(self, dim, dropout=0.1, name=None):
        super(LocalAggregator, self).__init__()
        self.dim = dim
        self.hidden = int(dim / 2)
        self.dropout = dropout

        self.a_0 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_1 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.a_3 = nn.Parameter(torch.Tensor(self.dim, 1))

        self.dp = nn.Dropout(dropout)
        self.bias = nn.Parameter(torch.Tensor(self.dim))
        self.linear = nn.Linear(2 * dim, dim)
        self.leakyrelu = nn.LeakyReLU(0.2)

    def forward(self, hidden, adj, mask, alpha_ent=1.1):
        h = hidden
        batch_size = h.shape[0]
        N = h.shape[1]
        mask_ = mask.unsqueeze(1).expand(-1, h.size(1), -1)

        a_input = (h.repeat(1, 1, N).view(batch_size, N * N, self.dim)
                   * h.repeat(1, N, 1)).view(batch_size, N, N, self.dim)

        e_0 = torch.matmul(a_input, self.a_0)
        e_1 = torch.matmul(a_input, self.a_1)
        e_2 = torch.matmul(a_input, self.a_2)
        e_3 = torch.matmul(a_input, self.a_3)

        e_0 = self.leakyrelu(e_0).squeeze(-1).view(batch_size, N, N)
        e_1 = self.leakyrelu(e_1).squeeze(-1).view(batch_size, N, N)
        e_2 = self.leakyrelu(e_2).squeeze(-1).view(batch_size, N, N)
        e_3 = self.leakyrelu(e_3).squeeze(-1).view(batch_size, N, N)

        mask = 0 * torch.ones_like(e_0)
        alpha = torch.where(adj.eq(1), e_0, mask)
        alpha = torch.where(adj.eq(2), e_1, alpha)
        alpha = torch.where(adj.eq(3), e_2, alpha)
        alpha = torch.where(adj.eq(4), e_3, alpha)
        alpha = alpha.masked_fill(mask_ == 0, -np.inf)
        alpha = entmax_bisect(alpha, alpha_ent, dim=-1)

        h = self.dp(h)
        output = torch.matmul(alpha, h)
        alpha = torch.sigmoid(self.linear(torch.cat((hidden, output), -1)))
        final_out = alpha * output + (1 - alpha) * hidden

        return final_out

class SideStructure(nn.Module):
    def __init__(self, dim, dp):
        super(SideStructure, self).__init__()
        self.dim = dim
        self.dp = dp
        self.FF2 = nn.Linear(self.dim, 1)

    def forward(self, x):
        out = self.FF2(x)
        alpha = torch.sigmoid(out) + 1
        return alpha


class SparseEdgeGNN(nn.Module):
    def __init__(self, dim, dp, heads):
        super(SparseEdgeGNN, self).__init__()
        self.dim = dim
        self.dp = dp
        self.heads = heads
        self.LAs = []
        self.fn1 = nn.Linear(self.dim * heads, self.dim * heads)
        self.fn2 = nn.Linear(self.dim * heads, self.dim)
        self.SideStructure = SideStructure(self.dim, self.dp)
        for i in range(self.heads):
            LA = LocalAggregator(self.dim, self.dp)
            self.add_module('local_aggregator_{}'.format(i), LA)
            self.LAs.append(LA)

    def forward(self, input1, adjF, mask):
        alpha_ent = self.SideStructure(input1)
        output = 0
        for i in range(self.heads):
            head_out = self.LAs[i](input1, adjF, mask, alpha_ent)
            output = output + head_out
        return output


class GEM(nn.Module):
    def __init__(self, dim):
        super(GEM, self).__init__()
        self.dim = dim
        self.w1 = nn.Linear(self.dim, self.dim, bias=False)
        self.w2 = nn.Linear(self.dim, self.dim, bias=False)

        self.w3 = nn.Linear(self.dim, self.dim, bias=False)
        self.w4 = nn.Linear(self.dim, self.dim, bias=False)

    def forward(self, sema_embed, topo_embed, mask):
        mask = mask.bool()
        sema_embed_ = sema_embed.clone()
        topo_embed_ = topo_embed.clone()
        alpha_left = self.w1(sema_embed_).unsqueeze(2)
        alpha_right = self.w2(topo_embed_).unsqueeze(3)
        alpha = torch.matmul(alpha_left, alpha_right) / math.sqrt(self.dim)
        alpha = alpha.squeeze().unsqueeze(2)
        sema_embed = sema_embed_ + alpha * (topo_embed_ - sema_embed_)

        beta_left = self.w3(topo_embed_).unsqueeze(2)
        beta_right = self.w4(sema_embed).unsqueeze(3)
        beta = torch.matmul(beta_left, beta_right) / math.sqrt(self.dim)
        beta = beta.squeeze().unsqueeze(2)
        topo_embed = topo_embed_ + beta * (sema_embed_ - topo_embed_)
        return sema_embed, topo_embed

class DenoiseEncoder(nn.Module):
    def __init__(self, opt, item_dim, pos_embedding, layers, dropout_in, dropout_hid):
        super(DenoiseEncoder, self).__init__()
        self.opt = opt
        self.heads = opt.heads
        self.dim = item_dim
        self.gnn_layers = layers[0]
        self.en_layers = layers[1]
        self.dpin = dropout_in
        self.dphid = dropout_hid
        self.pos_embedding = pos_embedding

        self.LocalEnds = []
        self.GlobalEnds = []
        self.DualEnhs = []
        for i in range(self.gnn_layers):
            LE = SparseEdgeGNN(self.dim, self.dphid, self.heads)
            GE = SparseEdgeGNN(self.dim, self.dphid, self.heads)
            self.add_module('local_encoder_{}'.format(i), LE)
            self.add_module('global_encoder_{}'.format(i), GE)
            self.LocalEnds.append(LE)
            self.GlobalEnds.append(GE)

        for i in range(self.en_layers):
            DE = GEM(self.dim)
            self.add_module('global_local_enhance_{}'.format(i), DE)
            self.DualEnhs.append(DE)

        self.w_1 = nn.Parameter(torch.Tensor(2 * self.dim, self.dim))
        self.w_2 = nn.Parameter(torch.Tensor(self.dim, 1))
        self.final_cat = nn.Linear(self.dim * 2, self.dim)
        self.cat = nn.Linear(self.dim * 2, self.dim, bias=False)
        self.fn = nn.Linear(self.dim * 2, 1)

        self.glu1 = nn.Linear(self.dim, self.dim)
        self.glu2 = nn.Linear(self.dim, self.dim, bias=False)
        self.dpin1 = nn.Dropout(self.dpin)
        self.dpin2 = nn.Dropout(self.dpin)

    def forward(self, gnn_input, drop_input, proto_input, adj, input_index, mask):
        adjD = adj
        gnn_input_mask = mask[0]
        gnn_item_mask = mask[1]
        gnn_mask = mask[2]
        gnn_item_mask_ori = mask[3]
        drop_mask = ~gnn_input_mask * gnn_item_mask_ori.unsqueeze(-1)
        len = gnn_input.shape[1]
        batch_size = gnn_input.shape[0]

        pos_emb = self.pos_embedding.weight[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim

        gnn_input = gnn_input * gnn_input_mask
        proto_input = proto_input * gnn_input_mask
        drop_input = drop_input * drop_mask

        # ----------------------- Backward Structure ----------------------- #
        avg = self.avgPool(gnn_input, gnn_item_mask).unsqueeze(1).repeat(1,drop_input.shape[1],1)
        alpha = self.fn(torch.cat((avg , drop_input), dim=-1))
        gnn_input = gnn_input + alpha * drop_input

        # ----------------------- Sparse-Edge GNNs (Feature-Level Denoise)----------------------- #
        local_emb, global_emb = self.gnn_encoder(gnn_input, proto_input, adjD, input_index, gnn_item_mask_ori, drop_mask)

        # ----------------------- SoftAttention ----------------------- #
        session_rep = self.SoftAtten(local_emb, global_emb, pos, gnn_mask)

        return session_rep

    def predict(self, gnn_input, drop_input, proto_embeds, adjD, input_index, mask):
        gnn_input_mask = mask[0]
        gnn_item_mask = mask[1]
        gnn_mask = mask[2]
        gnn_item_mask_ori = mask[3]
        drop_mask = ~gnn_input_mask * gnn_item_mask_ori.unsqueeze(-1)

        len = gnn_input.shape[1]
        batch_size = gnn_input.shape[0]
        pos_emb = self.pos_embedding.weight[:len]  # Seq_len x Embed_dim
        pos = pos_emb.unsqueeze(0).repeat(batch_size, 1, 1)  # Batch x Seq_len x Embed_dim

        gnn_input = gnn_input * gnn_input_mask
        drop_input = drop_input * drop_mask

        avg = self.avgPool(gnn_input, gnn_item_mask).unsqueeze(1).repeat(1,drop_input.shape[1],1)

        alpha = self.fn(torch.cat((avg , drop_input), dim=-1))
        gnn_input = gnn_input + alpha * drop_input

        local_emb, global_emb = self.gnn_encoder(gnn_input, proto_embeds, adjD, input_index, gnn_item_mask_ori, drop_mask)
        session_rep = self.SoftAtten(local_emb, global_emb, pos, gnn_mask)


        return session_rep

    def avgPool(self, input, mask):
        count = torch.sum(mask.float().unsqueeze(-1), 1)
        avg = torch.sum(input, 1) / (count + 1e-7)
        return avg

    def gnn_encoder(self, session_input, proto_input, adjD, inputs_index, gnn_mask, drop_mask):
        session_input = self.dpin2(session_input)
        proto_input = self.dpin2(proto_input)
        local_emb = session_input
        global_emb = proto_input
        local_emb_ = 0
        global_emb_ = 0
        for i in range(self.gnn_layers):
            local_emb = self.LocalEnds[i](local_emb, adjD, gnn_mask)
            global_emb = self.GlobalEnds[i](global_emb, adjD, gnn_mask)
            local_emb_ += local_emb
            global_emb_ += global_emb

        local_emb = local_emb_/(i+1)
        global_emb = global_emb_/(i+1)

        # ----------------------- Gloabl-Local Enhancement Module (GEM) ----------------------- #
        for i in range(self.en_layers):
            local_emb, global_emb = self.DualEnhs[i](local_emb, global_emb, gnn_mask)

        local_emb = local_emb[torch.arange(local_emb.size(0)).unsqueeze(1), inputs_index]
        global_emb = global_emb[torch.arange(global_emb.size(0)).unsqueeze(1), inputs_index]

        return local_emb, global_emb

    def SoftAtten(self, hidden, global_emb, pos, mask):
        mask = mask.float().unsqueeze(-1)  # Batch x Seq_len x 1
        hidden = hidden * mask

        lens = hidden.shape[1]  # Seq_len
        pos_emb = pos  # Batch x Seq_len x Embed_dim

        hs = torch.sum(hidden, -2) / (torch.sum(mask, 1) + 1e-7)
        hs = hs.unsqueeze(-2).repeat(1, lens, 1)

        ht = hidden[:,-1,:]

        nh = torch.matmul(torch.cat([pos_emb, hidden], -1), self.w_1)
        nh = torch.tanh(nh)

        beta = torch.sigmoid(self.glu1(nh) + self.glu2(hs))
        beta = torch.matmul(beta, self.w_2)
        beta = beta * mask
        select = torch.sum(beta * hidden, 1)

        return select


class HyPro(Module):
    def __init__(self, opt, item_to_topo, topo_to_items, num_node, max_len):
        super(HyPro, self).__init__()

        self.opt = opt
        self.item_dim = opt.embedding
        self.pos_dim = opt.posembedding
        self.batch_size = opt.batchSize

        self.num_node = num_node
        self.max_len = max_len
        self.temp = opt.temp
        self.theta = opt.theta

        self.item_to_topo = item_to_topo.cuda()
        self.topo_to_items = topo_to_items.cuda()

        self.dropout_in = opt.dropout_in
        self.dropout_hid = opt.dropout_hid

        self.layers = [opt.gnn_layer, opt.en_layer]
        self.item_centroids = None
        self.item_2cluster = None
        self.k = opt.k
        self.p = opt.p
        self.threshold = opt.threshold
        self.semantic_cluster = opt.semantic_cluster

        # embedding definition
        self.embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)
        self.pos_embedding = nn.Embedding(self.num_node, self.item_dim, max_norm=1.5)

        # component definition
        self.model = DenoiseEncoder(self.opt, self.item_dim, self.pos_embedding, self.layers, self.dropout_in,
                                    self.dropout_hid)
        self.Cl = ContrastiveLoss(self.batch_size, self.temp)

        # training definition
        self.loss_function = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=0)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step,
                                                         gamma=opt.lr_dc)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.item_dim)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, stdv)

    def forward(self, target, inputs, session_items, adj, inputs_index, step):

        # ------------------------------ Hybrid Prototypes ----------------------------------#
        if step % 5 == 0:
            if self.semantic_cluster == True:
                semnatic_embedding = self.seman_to_embed[self.item_to_seman]
            else:
                semnatic_embedding = self.embedding.weight.data
            self.hybrid_proto_embedding, self.proto_to_embed = self.hybrid_proto_acquire(semnatic_embedding)

        # ------------------------------ Prototypical Denoise (Item-level Denoise) -------------------------------------#
        session_item, drop_item, adj = self.denoise_module(inputs, adj, session_items)

        # ---------------------------------- Input and Mask --------------------------------------#
        gnn_seq = torch.ones_like(session_item)
        for i in range(len(session_item)):
            gnn_seq[i] = session_item[i][inputs_index[i]]
        gnn_seq_mask = (gnn_seq != 0).float()
        gnn_item_mask = (session_item != 0).float()
        gnn_item_mask_ori = (session_items != 0).float()

        session_embed = self.embedding.weight[session_item]
        drop_item_embed = self.embedding.weight[drop_item]
        hybrid_proto_embeds = self.hybrid_proto_embedding[session_item]
        target_proto = self.item_to_topo[target]

        timeline_mask = trans_to_cuda(torch.BoolTensor(session_item.detach().cpu().numpy() == 0))
        mask_crop = ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        mask = [mask_crop, gnn_item_mask, gnn_seq_mask, gnn_item_mask_ori]

        # ---------------------------------- Denoise Encoder --------------------------------------#
        output = self.model(session_embed, drop_item_embed, hybrid_proto_embeds, adj, inputs_index, mask)

        # ------------------------------------------Compute Score---------------------------------------------- #
        Result, session_rep = self.decoder(output)

        # ----------------------- Neighborhood Enhancement Task (NET) ----------------------- #
        cl = self.Cl(session_rep, target_proto)

        return Result, cl * self.theta

    def hybrid_proto_acquire(self, A):
        B = self.item_to_topo
        A = A.detach().cpu()
        B = B.detach().cpu().unsqueeze(-1)

        unique_B, inverse_indices = B.unique(return_inverse=True)

        sum_embeddings = torch.zeros(unique_B.size(0), A.size(1), device=A.device)
        sum_embeddings.index_add_(0, inverse_indices.squeeze(), A)
        counts = torch.bincount(inverse_indices.squeeze(), minlength=unique_B.size(0)).unsqueeze(1)
        mean_embeddings = sum_embeddings / counts

        # 5. 将平均值重新映射回原始矩阵
        result = mean_embeddings[inverse_indices.squeeze()]
        return trans_to_cuda(result), trans_to_cuda(mean_embeddings)

    def denoise_module(self, inputs, adj, items):

        item_2cluster = self.item_to_topo
        item_centroids = self.proto_to_embed
        threshold = self.threshold

        adj = adj.detach().cpu()
        inputs = inputs.detach().cpu()
        items = items.detach().cpu()
        p = 1 - self.p
        maskpad1 = (inputs != 0).cpu()
        maskpad2 = (items != 0).cpu()
        item_cat_seq = item_2cluster[inputs].detach().cpu() * maskpad1
        item_cat_items = item_2cluster[items].detach().cpu() * maskpad2
        seq_cat_embed =  item_centroids[item_cat_seq].detach().cpu() * maskpad1.unsqueeze(-1)
        item_cat_embed =  item_centroids[item_cat_items].detach().cpu() * maskpad2.unsqueeze(-1)

        mask = item_cat_seq != 0
        count = mask.sum(dim=1)
        count = count.unsqueeze(-1)
        avg_embed = torch.sum(seq_cat_embed, dim=1) / count
        target_embed = avg_embed
        target_embed = target_embed.unsqueeze(1).repeat(1, adj.shape[-1], 1)

        rand_matrix = torch.rand(item_cat_seq.shape[0], item_cat_seq.shape[1])

        x = F.normalize(item_cat_embed, p=2, dim=-1)
        y = F.normalize(target_embed, p=2, dim=-1)
        sim_matrix = torch.cosine_similarity(x, y, dim=2)
        mask_last = self.retain_last(inputs[:,-1].unsqueeze(-1), items)
        zeros = torch.zeros_like(mask_last)
        mask2 = 1 - torch.where(sim_matrix > threshold, 1, 0)
        mask3 = torch.where(mask2!=mask_last, zeros, mask2)

        mask_matrix_reverse = rand_matrix * mask3
        mask = 1 - torch.where(mask_matrix_reverse < p, 0, 1)

        mask_col = mask.unsqueeze(1).repeat(1, mask.shape[-1], 1)
        mask_row = mask.unsqueeze(-1).repeat(1, 1, mask.shape[-1])
        adj = adj * mask_col * mask_row
        item = items * mask
        drop_item = items * (1-mask)

        return trans_to_cuda(item), trans_to_cuda(drop_item), trans_to_cuda(adj)

    def retain_last(self, x, y):
        C = (y == x)  # C的形状为 (256, 39)，包含True/False值
        C_indices = C.float().argmax(dim=1, keepdim=True)  # (256, 1)
        mask = torch.ones_like(y)
        mask.scatter_(1, C_indices, 0)
        return mask

    def e_step(self):
        items_embedding = self.embedding.weight.detach().cpu().numpy()
        self.seman_to_embed, self.item_to_seman = self.run_kmeans(items_embedding[:])

    def run_kmeans(self, x1):
        # Semantic Clustering
        kmeans = faiss.Kmeans(d=x1.shape[-1], niter=50, k=self.k, gpu=True)
        kmeans.train(x1)
        cluster_cents = kmeans.centroids
        _, I = kmeans.index.search(x1, 1)

        centroids = trans_to_cuda(torch.Tensor(cluster_cents))
        centroids = F.normalize(centroids, p=2, dim=1)
        node2cluster = trans_to_cuda(torch.LongTensor(I).squeeze())

        return centroids, node2cluster

    def decoder(self, select):
        l_c = (select / torch.norm(select, dim=-1).unsqueeze(1))
        l_emb = self.embedding.weight[1:] / torch.norm(self.embedding.weight[1:], dim=-1).unsqueeze(1)
        z = 13 * torch.matmul(l_c, l_emb.t())

        return z, l_c

    def predict(self, data, k):
        # note that for prediction, CMCL don't have the decoupling module

        target, x_test, session_items, adj, inputs_index = data
        inputs = trans_to_cuda(x_test).long()
        adj = trans_to_cuda(adj).long()
        session_items = trans_to_cuda(session_items).long()

        session_item, drop_item, adj = self.denoise_module(inputs, adj, session_items)

        # session = session_items
        gnn_seq = torch.ones_like(session_item)
        for i in range(len(session_item)):
            gnn_seq[i] = session_item[i][inputs_index[i]]
        gnn_seq_mask = (gnn_seq != 0).float()
        gnn_item_mask = (session_item != 0).float()
        gnn_item_mask_ori = (session_items != 0).float()

        timeline_mask = trans_to_cuda(torch.BoolTensor(session_item.detach().cpu().numpy() == 0))
        mask_crop = ~timeline_mask.unsqueeze(-1)  # broadcast in last dim

        session_embed = self.embedding.weight[session_item]
        drop_item_embed = self.embedding.weight[drop_item]
        hybrid_proto_embeds = self.hybrid_proto_embedding[session_item]

        mask = [mask_crop, gnn_item_mask, gnn_seq_mask, gnn_item_mask_ori]

        output = self.model.predict(session_embed, drop_item_embed, hybrid_proto_embeds, adj,inputs_index, mask)
        result1, session_rep = self.decoder(output)
        rank1 = torch.argsort(result1, dim=1, descending=True)
        return rank1[:, 0:k], output


def trans_to_cuda(variable):
    if torch.cuda.is_available():
        return variable.cuda()
        # return variable
    else:
        return variable


def forward(model, data, step):
    target, u_input, session_items, adj, u_input_index = data

    session_items = trans_to_cuda(session_items).long()
    u_input = trans_to_cuda(u_input).long()
    target = trans_to_cuda(target).long()
    adj = trans_to_cuda(adj).long()
    Result, cl = model(target, u_input, session_items, adj, u_input_index, step)
    return Result, cl
