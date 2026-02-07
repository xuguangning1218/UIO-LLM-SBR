import torch
import torch.nn.functional as F

class ContrastiveLearningModel(torch.nn.Module):
    def __init__(self, temperature=0.5):
        super(ContrastiveLearningModel, self).__init__()
        self.temperature = temperature

    def forward(self, sess_final, l_c):
        # 对 sess_final 和 l_c 进行 L2 归一化
        sess_final = F.normalize(sess_final, dim=-1, p=2)
        l_c = F.normalize(l_c, dim=-1, p=2)
        
        # 计算正样本相似性
        positive_similarity = torch.sum(sess_final * l_c, dim=-1)  # [b]

        # 构建负样本：通过打乱 l_c 的顺序来生成负样本
        batch_size = sess_final.size(0)
        neg_indices = torch.randperm(batch_size)
        while torch.any(neg_indices == torch.arange(batch_size)):
            neg_indices = torch.randperm(batch_size)
        
        negative_similarity = torch.sum(sess_final * l_c[neg_indices], dim=-1)  # [b]

        # 计算对比学习损失
        positive_loss = -F.logsigmoid(positive_similarity / self.temperature)
        negative_loss = -F.logsigmoid(-negative_similarity / self.temperature)
        contrastive_loss = (positive_loss + negative_loss).mean()

        return contrastive_loss

# # 在训练过程中使用对比学习
# def train_batch(model, batch_data):
#     # 计算当前会话的 sess_final 和 l_c
#     sess_final = model.compute_sess_final(batch_data)
#     l_c = model.compute_l_c(batch_data)
    
#     # 计算对比学习损失
#     contrastive_loss = model.contrastive_learning_model(sess_final, l_c)
    
#     # 计算主要任务的损失
#     main_loss = compute_main_loss(model, batch_data)
    
#     # 合并损失
#     total_loss = main_loss + contrastive_loss_weight * contrastive_loss
#     total_loss.backward()
#     optimizer.step()
