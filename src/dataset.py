"""
Script to create the Dataset for the GAN.
"""

import torch
from torch.utils.data import Dataset

class PasswordDataset(Dataset):
    """Dataset for tokenized passwords."""

    def __init__(self, passwords, stoi, seq_len):
        """
        Args:
            passwords: List of password strings.
            stoi: Dictionary mapping characters to indices.
            seq_len: Maximum sequence length.
        """
        self.passwords = passwords
        self.stoi = stoi
        self.seq_len = seq_len

    def __len__(self):
        """Return number of passwords."""
        return len(self.passwords)

    def encode(self, pwd):
        """Convert a password string to a tensor of token ids."""
        ids = [self.stoi.get(c, 1) for c in pwd]
        ids = ids[:self.seq_len]
        ids += [0] * (self.seq_len - len(ids))  # pad
        return torch.tensor(ids, dtype=torch.long)

    def __getitem__(self, index):
        """Return tokenized password at the given index."""
        return self.encode(self.passwords[index])