import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from data.cifar import CIFAR10Dataset  # 假设 data/cifar.py 定义了 CIFAR 数据加载逻辑
from data.stl import STL10Dataset  # 假设 data/stl.py 定义了 STL 数据加载逻辑
import torchvision.utils as vutils

# ==============================
# 1. 设置设备和路径
# ==============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_root = "../data"  # 假设 data 文件夹与 experiment 同级


# ==============================
# 2. 加载数据
# ==============================
def load_data(dataset_name, batch_size=64):
    if dataset_name == "cifar10":
        # CIFAR-10 数据集加载
        dataset = CIFAR10Dataset(root=os.path.join(data_root, "wnx/cifar-10-python"), train=True)
    elif dataset_name == "stl10":
        # STL-10 数据集加载
        dataset = STL10Dataset(root=os.path.join(data_root, "wnx/stl10_binary"), train=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    return dataloader


# ==============================
# 3. 定义扩散模型
# ==============================
class DiffusionModel:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        """添加噪声"""
        noise = torch.randn_like(x0).to(device)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def reverse_process(self, model, shape):
        """从噪声逐步生成样本"""
        x = torch.randn(shape).to(device)  # 随机初始化噪声
        for t in reversed(range(self.num_steps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt()
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t).sqrt()
            x = (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t
        return x


# ==============================
# 4. 定义生成模型（U-Net）
# ==============================
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1, 1, 1, 1).float()  # 时间步嵌入
        x = torch.cat([x, t_emb.repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = self.conv4(x)
        return x


# ==============================
# 5. 训练扩散模型
# ==============================
def train_diffusion_model(dataset_name, epochs=5, batch_size=64):
    # 加载数据
    dataloader = load_data(dataset_name, batch_size)

    # 初始化模型和扩散过程
    model = SimpleUNet().to(device)
    diffusion = DiffusionModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # 训练过程
    for epoch in range(epochs):
        for i, (x0, _) in enumerate(dataloader):
            x0 = x0.to(device)
            t = torch.randint(0, diffusion.num_steps, (x0.size(0),), device=device)
            xt, noise = diffusion.add_noise(x0, t)
            predicted_noise = model(xt, t)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 100 == 0:
                print(f"Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}")

    return model, diffusion


# ==============================
# 6. 生成样本
# ==============================
def generate_samples(model, diffusion, num_samples=64):
    generated_samples = diffusion.reverse_process(model, (num_samples, 3, 32, 32))
    return generated_samples


# ==============================
# 7. 主函数
# ==============================
if __name__ == "__main__":
    # 选择数据集
    dataset_name = "cifar10"  # 或 "stl10"

    # 训练扩散模型
    model, diffusion = train_diffusion_model(dataset_name)

    # 生成样本
    samples = generate_samples(model, diffusion)

    # 可视化生成的样本
    vutils.save_image(samples, f"{dataset_name}_generated_samples.png", nrow=8, normalize=True)
    print(f"Generated samples saved to {dataset_name}_generated_samples.png")
