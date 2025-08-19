import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import os
from sklearn.decomposition import PCA
import random


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
checkpoint_dir = "/exp_data/sjx/star/gan_data/checkpoints/"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint_dir = "/exp_data/sjx/star/gan_data/checkpoints"
epoch300_path = os.path.join(checkpoint_dir, "generator_epoch300.pt")
best_path = os.path.join(checkpoint_dir, "best_generator.pt")

if os.path.exists(epoch300_path) and os.path.exists(best_path):
    print(" A trained model has been detected, skipping the training phase.")
    skip_training = True
else:
    skip_training = False

# ========================
# Generator with Conv1d
# ========================
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(noise_dim, seq_len * 256)
        self.conv = nn.Sequential(
            nn.Conv1d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, embed_dim, 3, padding=1),
            nn.Tanh()
        )

    def forward(self, z):
        x = self.fc(z).view(-1, 256, seq_len)  # (B, 256, 300)
        x = self.conv(x)  # (B, embed_dim, 300)
        return x.transpose(1, 2)  # (B, 300, embed_dim)

# ========================
# Discriminator with Conv1d
# ========================
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(embed_dim, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * seq_len, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (B, embed_dim, 300)
        x = self.conv(x)       # (B, 128, 300)
        x = x.reshape(x.size(0), -1)  # (B, 128*300)
        return self.fc(x)

# =======================
# Gradient Penalty
# ========================
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
# =======================
# Load Real Data
# ========================
real_data = np.load("/exp_data/sjx/star/first_data/ESM-embedding/negative_all_embedding.npy")
print("real data shape:", real_data.shape)
print("range of real data：", real_data.min(), real_data.max())
real_tensor = torch.tensor(real_data, dtype=torch.float32)
dataloader = DataLoader(TensorDataset(real_tensor), batch_size=batch_size, shuffle=True)

# ========================
# Models
# ========================
G = Generator().to(device)
D = Discriminator().to(device)
optimizer_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
optimizer_D = optim.Adam(D.parameters(), lr=1e-4, betas=(0.5, 0.9))

# =======================
# Training
# ========================
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

        
        if epoch % 10 == 0 or epoch == epochs:
            save_path = os.path.join(checkpoint_dir, f"generator_epoch{epoch}.pt")
            torch.save(G.state_dict(), save_path)
            print(f"[Checkpoint] Saved generator to {save_path}")

        
        if g_loss.item() < best_g_loss:
            best_g_loss = g_loss.item()
            best_path = os.path.join(checkpoint_dir, "best_generator.pt")
            torch.save(G.state_dict(), best_path)
            print(f"[BEST] Saved best generator with G loss = {best_g_loss:.4f}")

# =======================
# Save Model & Generate
# ========================
# ========================

# ========================
print("\nLoading best generator for data generation...")


G.load_state_dict(torch.load(os.path.join(checkpoint_dir, "generator_epoch260.pt")))
G.eval()


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
save_data_path = "/exp_data/sjx/star/gan_data/260_e_fake_negative_embeddings.npy"
np.save(save_data_path, generated)
print(f"Saved 2,435 generated data entries: {save_data_path}")
