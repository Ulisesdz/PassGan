"""
Script for some utility functions.
"""

import os
import random

import torch
import numpy as np
import matplotlib.pyplot as plt


def set_seed(seed: int) -> None:
    """
    This function sets a seed and ensure a deterministic behavior.

    Args:
        seed : Seed number to fix randomness.
    """

    # set seed in numpy and random
    np.random.seed(seed)
    random.seed(seed)

    # set seed and deterministic algorithms for torch
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    # Ensure all operations are deterministic on GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # for deterministic behavior on cuda >= 10.2
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def decode_tokens(token_batch: torch.Tensor, itos: dict = None) -> list[str]:
    """
    Converts a batch of token sequences into a list of decoded strings.

    Args:
        token_batch (torch.Tensor): Tensor of shape [batch, seq_len, vocab_size]
            containing logits or probabilities over characters.
        itos (dict, optional): Dictionary mapping indices to characters.

    Returns:
        List[str]: List of decoded password strings.
    """

    if itos is None:
        alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!@#$%^&*()-_=+?"
        itos = {i+1: c for i, c in enumerate(alphabet)}
        itos[0] = ""  # padding

    # token_batch: [batch, seq_len, vocab_size] or [batch, seq_len] if argmax already
    if token_batch.dim() == 3:  # logits -> indices
        token_ids = token_batch.argmax(dim=-1).detach().cpu().numpy()
    else:
        token_ids = token_batch.detach().cpu().numpy()

    decoded = []
    for seq in token_ids:
        chars = [itos.get(int(idx), "") for idx in seq if int(idx) != 0]  # ignore padding
        decoded.append("".join(chars))
    return decoded


def save_generated_samples(
    epochs: list[int],
    samples: list[list[str]],
    samples_per_epoch: int = 8,
    path: str = "generated_samples.txt",
) -> None:
    """
    Saves generated samples across epochs in a text file.

    Args:
        epochs (list[int]): List of epoch numbers.
        samples (list[list[str]]): List in which each element is a list of generated
        passwords (decoded strings).
        samples_per_epoch (int, optional): Number of samples to save per epoch.
        path (str, optional): Output file path.

    Returns:
        None
    """

    with open(path, "w", encoding="utf-8") as f:
        for i, epoch in enumerate(epochs):
            f.write(f"===== Epoch {epoch} =====\n")
            epoch_samples = samples[i][:samples_per_epoch]

            for pwd in epoch_samples:
                f.write(pwd + "\n")

            f.write("\n")

    print(f"[INFO] Saved generated samples to {path}")


def save_gan_losses(
    d_real_losses: list[float],
    d_fake_losses: list[float],
    d_losses: list[float],
    g_losses: list[float],
    path: str,
) -> None:
    """Saves the evolution of the losses in the GAN.

    Args:
        d_real_losses (List[float]): Losses of the discriminator when
        predicting real samples.
        d_fake_losses (List[float]): Losses of the discriminator when
        predicting fake samples.
        d_losses (List[float]): Combined discriminator losses.
        g_losses (List[float]): Generator losses.
        path (str): Path to save the generated plot.
    """

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    assert isinstance(axs, np.ndarray)

    axs[0].plot(d_real_losses, linewidth=0.5, label="Real")
    axs[0].plot(d_fake_losses, linewidth=0.5, label="Fake")
    axs[0].plot(d_losses, linewidth=0.5, label="Total")
    axs[0].set_xlabel("Iteration")
    axs[0].set_ylabel("Loss")
    axs[0].set_title("Discriminator")
    axs[0].legend()

    axs[1].plot(g_losses)
    axs[1].set_xlabel("Iteration")
    axs[1].set_ylabel("Loss")
    axs[1].set_title("Generator")

    fig.savefig(path)
    plt.close(fig)
    print(f"[INFO] Saved GAN loss curves at {path}")


def get_noise(n_samples: int, z_dim: int, device: str = "cpu") -> torch.Tensor:
    """
    Function for creating noise vectors: Given the dimensions (n_samples, z_dim),
    creates a tensor of that shape filled with random numbers from the normal distr.

    Parameters:
        n_samples: the number of samples to generate, a scalar
        z_dim: the dimension of the noise vector, a scalar
        device: the device type
    """
    return torch.randn((n_samples, z_dim), device=device)