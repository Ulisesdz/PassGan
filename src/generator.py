"""
Script to generate passwords from a trained GAN (PassGAN-like).
"""

import torch
from src.model import Generator
from src.utils import get_noise, decode_tokens

# ---------------------------------------------------------
#               CONFIGURATION
# ---------------------------------------------------------

MODEL_PATH = "results/generator.pt"
Z_DIM = 128
HIDDEN_DIM = 256
SEQ_LEN = 12
NUM_LAYERS = 2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_SAMPLES = 20  # Número de contraseñas a generar

# Define alphabet and mappings exactly as in training
alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+?"
stoi = {c: i + 1 for i, c in enumerate(alphabet)}
stoi["<pad>"] = 0
VOCAB_SIZE = len(stoi)
print(VOCAB_SIZE)
itos = {i: c for c, i in stoi.items()}

# ---------------------------------------------------------
#               INITIALIZE GENERATOR
# ---------------------------------------------------------

generator = Generator(
    z_dim=Z_DIM,
    hidden_dim=HIDDEN_DIM,
    seq_len=SEQ_LEN,
    vocab_size=VOCAB_SIZE,
    num_layers=NUM_LAYERS,
).to(DEVICE)

# Load trained weights
generator.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
generator.eval()

# ---------------------------------------------------------
#               GENERATE PASSWORDS
# ---------------------------------------------------------

with torch.no_grad():
    noise = get_noise(NUM_SAMPLES, Z_DIM, DEVICE)
    logits = generator(noise)  # [batch, seq_len, vocab_size]
    # Muestreo probabilístico en lugar de argmax
    probs = torch.softmax(logits, dim=-1)
    tokens = torch.multinomial(probs.view(-1, VOCAB_SIZE), 1).view(NUM_SAMPLES, SEQ_LEN)
    passwords = decode_tokens(tokens, itos)

# Print generated passwords
print("Generated passwords:")
for i, pwd in enumerate(passwords, 1):
    print(f"{i}: {pwd}")
