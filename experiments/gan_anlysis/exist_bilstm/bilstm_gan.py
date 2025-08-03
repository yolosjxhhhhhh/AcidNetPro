import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import os
import random

# 配置
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seq_len = 300
embed_dim = 1152
noise_dim = 128
batch_size = 16
n_critic = 5
lambda_gp = 10
epochs = 300
checkpoint_dir = "/exp_data/sjx/star/experiments/gan_anlysis/data/exist_bilstm_data/chekpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

epoch300_path = os.path.join(checkpoint_dir, "generator_epoch300.pt")
best_path = os.path.join(checkpoint_dir, "best_generator.pt")
if os.path.exists(best_path):
    print("[✓] 已检测到已完成训练的模型，将跳过训练阶段")
    skip_training = True
else:
    skip_training = False


class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(noise_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, seq_len * embed_dim),
            nn.Tanh()
        )
    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, seq_len, embed_dim)  # (B, 300, 1152)
        return x
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(seq_len * embed_dim, 2048),
            nn.LeakyReLU(0.2),
            nn.Linear(2048, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1)
        )
    def forward(self, x):
        x = x.view(x.size(0), -1)  # (B, 300*1152)
        return self.fc(x)
def compute_gradient_penalty(D, real_samples, fake_samples):
    alpha = torch.rand(real_samples.size(0), 1, 1, device=device)
    alpha = alpha.expand_as(real_samples)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones(d_interpolates.size(), device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    return ((gradients.norm(2, dim=1) - 1) ** 2).mean()
real_data = np.load("/exp_data/sjx/star/first_data/ESM-embedding/negative_all_embedding.npy")
print("真实数据 shape:", real_data.shape)
print("真实数据值域：", real_data.min(), real_data.max())
real_tensor = torch.tensor(real_data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(real_tensor), batch_size=batch_size, shuffle=True)
G = Generator().to(device)
D = Discriminator().to(device)
optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))
best_g_loss = float("inf")
if not skip_training:
    for epoch in range(1, epochs + 1):
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for i, (real,) in enumerate(pbar):
            real = real.to(device)

            # === Train D ===
            for _ in range(n_critic):
                z = torch.randn(real.size(0), noise_dim).to(device)
                fake = G(z).detach()
                real_score = D(real)
                fake_score = D(fake)
                gp = compute_gradient_penalty(D, real, fake)
                d_loss = -torch.mean(real_score) + torch.mean(fake_score) + lambda_gp * gp
                optimizer_D.zero_grad()
                d_loss.backward()
                optimizer_D.step()

            # === Train G ===
            z = torch.randn(real.size(0), noise_dim).to(device)
            fake = G(z)
            g_loss = -torch.mean(D(fake))
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            pbar.set_postfix({
                "D loss": f"{d_loss.item():.2f}",
                "G loss": f"{g_loss.item():.2f}"
            })

        # === 每 10 epoch 或最后一个 epoch保存模型 ===
        if epoch % 100 == 0 or epoch == epochs:
            save_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}.pt")
            torch.save(G.state_dict(), save_path)
            print(f"[Checkpoint] Saved generator to {save_path}")

        # === 保存表现最好的 G ===
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            best_path = os.path.join(checkpoint_dir, "best_generator.pt")
            torch.save(G.state_dict(), best_path)
            print(f"[BEST] Saved best generator with G loss = {best_g_loss:.4f}")
print("\nLoading best generator for data generation...")

# 重新加载 best generator 权重
G.load_state_dict(torch.load(os.path.join(checkpoint_dir, "best_generator.pt")))
G.eval()

# 生成 2435 条数据
gen_total = 2435
batch_size = 256
generated = []

with torch.no_grad():
    total = 0
    while total < gen_total:
        current_batch = min(batch_size, gen_total - total)
        z = torch.randn(current_batch, noise_dim).to(device)
        fake = G(z).cpu().numpy()
        generated.append(fake)
        total += current_batch

generated = np.concatenate(generated, axis=0)
save_data_path = "/exp_data/sjx/star/experiments/gan_anlysis/data/exist_bilstm_data/fc_only_fake_negative_embeddings.npy"
np.save(save_data_path, generated)
print(f"已保存生成的 2435 条数据到: {save_data_path}")