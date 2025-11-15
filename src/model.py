"""
Script to create the Discriminator and Generator for the GAN.
"""

from torch import nn
import torch
import torch.nn.functional as F


class Generator(nn.Module):
    """Generator mapping noise vectors to sequences of character logits."""

    def __init__(
        self,
        z_dim: int,
        hidden_dim: int,
        seq_len: int,
        vocab_size: int,
        num_layers: int = 2,
    ):
        """
        Args:
            z_dim: Dimension of input noise vector.
            hidden_dim: Hidden size for LSTM and linear layers.
            seq_len: Length of output sequence.
            vocab_size: Number of possible characters.
            num_layers: Number of LSTM layers.
        """
        super().__init__()
        self.seq_len = seq_len
        self.vocab_size = vocab_size

        self.fc = nn.Linear(z_dim, hidden_dim)
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,  # Regularization for deeper LSTMs
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, noise: torch.Tensor, tau: float = 1.0, hard: bool = False) -> torch.Tensor:
        """
        Forward pass: noise -> sequence logits or sampled token probabilities.
        
        Uses Gumbel-Softmax trick during training to allow gradient propagation
        through discrete sampling, increasing diversity in generated sequences.

        Args:
            noise: Tensor of shape [batch_size, z_dim].
            tau: Temperature parameter for Gumbel-Softmax.
            hard: If True, use hard one-hot sampling; otherwise soft.
        Returns:
            Tensor of logits (eval) or sampled distributions (train).
        """
        x = torch.relu(self.fc(noise))
        x = x.unsqueeze(1).repeat(1, self.seq_len, 1)  # Repeat noise across sequence length
        x, _ = self.lstm(x)
        x = self.layer_norm(x)
        logits = self.fc_out(x)

        if self.training:
            # Gumbel-Softmax allows differentiable sampling of discrete tokens
            y = F.gumbel_softmax(logits, tau=tau, hard=hard)
            return y
        else:
            # During evaluation, select most likely token per position
            return logits.float() 


class Discriminator(nn.Module):
    """Discriminator classifying sequences as real or fake."""

    def __init__(
        self,
        seq_len: int,
        vocab_size: int,
        channel_dim: int = 64,
    ):
        """
        Args:
            seq_len: Length of input sequence.
            vocab_size: Number of possible characters.
            channel_dim: Dimension of embedding channels.
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, channel_dim)

        # Using multiple Conv1D layers to capture local sequential patterns
        self.conv = nn.Sequential(
            nn.Conv1d(channel_dim, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, 3, padding=1),
            nn.LeakyReLU(0.2),
        )

        # Fully connected layers with dropout for regularization
        self.fc = nn.Sequential(
            nn.Linear(512 * seq_len, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: sequences -> probability of being real.
        
        Args:
            x: Tensor of shape [batch_size, seq_len] containing token indices.
        Returns:
            Tensor of shape [batch_size, 1] with probability of being real.
        """
        x = self.embedding(x)          # Shape: [batch, seq_len, channel_dim]
        x = x.transpose(1, 2)          # Shape: [batch, channel_dim, seq_len] for Conv1d
        features = self.conv(x)
        features = features.reshape(x.size(0), -1)
        return self.fc(features)