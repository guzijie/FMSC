import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import inception_v3
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.cluster import KMeans
from tqdm import tqdm

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =====================================
# 1. Load CIFAR-10 Dataset
# =====================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load CIFAR-10
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=64, shuffle=True, num_workers=4)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=4)


# =====================================
# 2. Define Diffusion Model Framework
# =====================================
class DiffusionModel:
    def __init__(self, num_steps=1000, beta_start=1e-4, beta_end=0.02):
        self.num_steps = num_steps
        self.betas = torch.linspace(beta_start, beta_end, num_steps).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)

    def add_noise(self, x0, t):
        noise = torch.randn_like(x0).to(device)
        sqrt_alpha_cumprod = self.alpha_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod = (1 - self.alpha_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        xt = sqrt_alpha_cumprod * x0 + sqrt_one_minus_alpha_cumprod * noise
        return xt, noise

    def sample(self, model, shape):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.num_steps)):
            # Predict noise at time t
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_tensor)

            # Update x
            beta_t = self.betas[t]
            alpha_t = self.alphas[t]
            alpha_cumprod_t = self.alpha_cumprod[t]
            sqrt_alpha_cumprod_t = alpha_cumprod_t.sqrt()
            sqrt_one_minus_alpha_cumprod_t = (1 - alpha_cumprod_t).sqrt()
            x = (x - beta_t / sqrt_one_minus_alpha_cumprod_t * predicted_noise) / sqrt_alpha_cumprod_t

        return x


# =====================================
# 3. Define Simple U-Net for Diffusion Model
# =====================================
class SimpleUNet(torch.nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = torch.nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = torch.nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv4 = torch.nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = t.view(-1, 1, 1, 1).float()  # Embed time as a scalar
        x = torch.cat([x, t_emb.repeat(1, 1, x.size(-2), x.size(-1))], dim=1)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)
        return x


# =====================================
# 4. Train Diffusion Model
# =====================================
# Initialize model and diffusion process
model = SimpleUNet().to(device)
diffusion = DiffusionModel()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 10
for epoch in range(epochs):
    for batch in tqdm(trainloader):
        x0, _ = batch
        x0 = x0.to(device)

        # Sample random time steps
        t = torch.randint(0, diffusion.num_steps, (x0.size(0),), device=device)
        xt, noise = diffusion.add_noise(x0, t)

        # Predict noise
        predicted_noise = model(xt, t)
        loss = F.mse_loss(predicted_noise, noise)

        # Optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# =====================================
# 5. Generate Samples
# =====================================
# Generate 100 samples
generated_samples = diffusion.sample(model, shape=(100, 3, 32, 32))

# Save generated samples
torchvision.utils.save_image(generated_samples, "generated_samples.png", nrow=10)

# =====================================
# 6. Evaluate FID and IS
# =====================================
# Placeholder: Implement FID and IS calculation here
# You can use pre-trained Inception-v3 for feature extraction
