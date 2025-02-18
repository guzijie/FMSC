import torch
import torch.nn as nn
import torch.optim as optim


# 假设我们有一个预训练的 score network，它是一个估计数据分布梯度的网络
class SimpleScoreNetwork(nn.Module):
    def __init__(self, input_dim):
        super(SimpleScoreNetwork, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        return self.fc(x)


# 加噪过程
def add_noise(samples, sigma):
    noise = torch.randn_like(samples) * sigma
    return samples + noise


# 生成新样本特征的函数
def generate_samples_with_diffusion(score_network, input_features, n_steps=100, step_size=0.01, sigma=0.5):
    # 输入：score_network - 扩散模型中的得分网络
    #      input_features - 原始样本特征
    #      n_steps - 去噪步数
    #      step_size - 每一步的步长（学习率）
    #      sigma - 加噪声的强度
    # 输出：生成的新样本特征

    # 初始化样本特征，加初始噪声
    x = add_noise(input_features, sigma)

    # 执行扩散过程，逐步去噪
    for step in range(n_steps):
        # 计算得分（梯度）
        x.requires_grad_(True)
        score = score_network(x)

        # 更新样本特征，向去噪方向移动
        with torch.no_grad():
            grad = score - x  # 得分网络给出的去噪方向
            x = x + step_size * grad  # 更新样本特征
            noise = torch.randn_like(x) * (sigma * (1 - step / n_steps))  # 逐步减小的噪声
            x = x + noise  # 加入少量噪声

    return x.detach()


# 示例使用
# input_dim = 32  # 假设输入特征是 32 维
# score_network = SimpleScoreNetwork(input_dim)  # 初始化一个简单的得分网络
# input_features = torch.randn(5, input_dim)  # 生成一些随机特征作为输入
#
# generated_features = generate_samples_with_diffusion(score_network, input_features)
# print("生成的新样本特征：", generated_features)
