"""
Improved PassGAN-like GAN training script.
Uses Gumbel-Softmax in generator for diverse outputs, label smoothing,
and separate learning rates for stability.
"""

import os
import torch
from torch.utils.data import DataLoader
from torch import nn, optim

from src.model import Generator, Discriminator
from src.dataset import PasswordDataset
from src.utils import (
    set_seed,
    get_noise,
    decode_tokens,
    save_generated_samples,
    save_gan_losses,
)

# ---------------------------------------------------------
# HYPERPARAMETERS
# ---------------------------------------------------------
DATA_PATH = "data/PasswordDictionary.txt"
SEQ_LEN = 12
BATCH_SIZE = 64
Z_DIM = 128
HIDDEN_DIM = 256
GEN_LAYERS = 2
CHANNELS = 64
LR_GEN = 1e-4
LR_DISC = 5e-5
EPOCHS = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_SAMPLES_EVERY = 5
GUMBEL_TAU = 1.0  # temperature for Gumbel-Softmax

set_seed(42)

# ---------------------------------------------------------
# LOAD DATASET AND BUILD VOCAB
# ---------------------------------------------------------
def load_passwords(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"[ERROR] Dataset file '{path}' not found.")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines() if l.strip()]
    return lines

alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+?"
stoi = {c: i + 1 for i, c in enumerate(alphabet)}
stoi["<pad>"] = 0
itos = {i: c for c, i in stoi.items()}

password_list = load_passwords(DATA_PATH)
dataset = PasswordDataset(password_list, stoi, SEQ_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# ---------------------------------------------------------
# INITIALIZE MODELS AND OPTIMIZERS
# ---------------------------------------------------------
generator = Generator(
    z_dim=Z_DIM,
    hidden_dim=HIDDEN_DIM,
    seq_len=SEQ_LEN,
    vocab_size=len(stoi),
    num_layers=GEN_LAYERS,
).to(DEVICE)

discriminator = Discriminator(
    seq_len=SEQ_LEN,
    vocab_size=len(stoi),
    channel_dim=CHANNELS,
).to(DEVICE)

criterion = nn.BCELoss()
g_opt = optim.Adam(generator.parameters(), lr=LR_GEN, betas=(0.5, 0.999))
d_opt = optim.Adam(discriminator.parameters(), lr=LR_DISC, betas=(0.5, 0.999))

# ---------------------------------------------------------
# TRAINING LOOP
# ---------------------------------------------------------
d_real_losses, d_fake_losses, d_losses, g_losses = [], [], [], []
saved_epochs, saved_samples = [], []

print("[INFO] Starting training...")

for epoch in range(1, EPOCHS + 1):
    for real in dataloader:
        real = real.to(DEVICE)
        batch_size = real.size(0)

        # -----------------------------
        # TRAIN DISCRIMINATOR
        # -----------------------------
        d_opt.zero_grad()

        # Label smoothing for real labels
        real_labels = torch.ones((batch_size, 1), device=DEVICE) * 0.9
        fake_labels = torch.zeros((batch_size, 1), device=DEVICE)

        # Real samples
        preds_real = discriminator(real)
        loss_real = criterion(preds_real, real_labels)

        # Fake samples (using Gumbel-Softmax)
        noise = get_noise(batch_size, Z_DIM, DEVICE)
        fake_logits = generator(noise, tau=GUMBEL_TAU, hard=False)  # soft sample
        fake_tokens = fake_logits.argmax(dim=-1)
        preds_fake = discriminator(fake_tokens.detach())
        loss_fake = criterion(preds_fake, fake_labels)

        d_loss = loss_real + loss_fake
        d_loss.backward()
        d_opt.step()

        # -----------------------------
        # TRAIN GENERATOR
        # -----------------------------
        g_opt.zero_grad()

        noise = get_noise(batch_size, Z_DIM, DEVICE)
        fake_logits = generator(noise, tau=GUMBEL_TAU, hard=False)
        fake_tokens = fake_logits.argmax(dim=-1)
        preds = discriminator(fake_tokens)

        g_loss = criterion(preds, torch.ones_like(preds))  # fool discriminator
        g_loss.backward()
        g_opt.step()

        d_real_losses.append(loss_real.item())
        d_fake_losses.append(loss_fake.item())
        d_losses.append(d_loss.item())
        g_losses.append(g_loss.item())

    print(f"[EPOCH {epoch}/{EPOCHS}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")

    # Save samples every few epochs
    if epoch % SAVE_SAMPLES_EVERY == 0:
        with torch.no_grad():
            noise = get_noise(16, Z_DIM, DEVICE)
            logits = generator(noise, tau=GUMBEL_TAU, hard=False)
            decoded = decode_tokens(logits, itos)

        saved_epochs.append(epoch)
        saved_samples.append(decoded)

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------
os.makedirs("results", exist_ok=True)

save_generated_samples(
    epochs=saved_epochs,
    samples=saved_samples,
    path="results/generated_samples.txt",
)

save_gan_losses(
    d_real_losses=d_real_losses,
    d_fake_losses=d_fake_losses,
    d_losses=d_losses,
    g_losses=g_losses,
    path="results/loss_curves.png",
)

torch.save(generator.state_dict(), "results/generator.pt")
torch.save(discriminator.state_dict(), "results/discriminator.pt")

print("[INFO] Training completed.")
print("[INFO] Saved model and logs in /results/")