import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import time

class STEWeights(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.threshold = 0.33 # For ternary SAE
        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, x):
        # FP:
        # return F.linear(x, self.weight)

        # Ternary:
        sign_weight = torch.sign(self.weight)
        mask = (torch.abs(self.weight) >= self.threshold).float()
        hard_weights = sign_weight * mask # Ternary SAE
        # hard_weights = torch.sign(self.weight)  # Binary
        return F.linear(x, self.weight + (hard_weights - self.weight).detach())

class TernarySparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = STEWeights(hidden_dim, input_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        return h, self.decoder(h)

class HiddenStatesTorchDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        file_paths: .pt file path. The file is expected to store a tensor 
                    with shape of (num_contexts, tokens_per_context, feature_dim) where
                    feature_dim is expected to be 512.
        transform: Optional transformation to apply to each sample.
        """
        self.data = torch.load(file_path, map_location='cpu')
        self.transform = transform
        num_contexts, tokens_per_context, feature_dim = self.data.shape
        
        self.cum_sizes = num_contexts * tokens_per_context
        self.files_info = (file_path, num_contexts, tokens_per_context, feature_dim)

    def __len__(self):
        return self.cum_sizes

    def __getitem__(self, idx):
        # Map the local index to (context index, token index)
        context_idx = idx // self.files_info[2]
        token_idx = idx % self.files_info[2] 
        sample = self.data[context_idx, token_idx, :]  # Each sample is a 512-d tensor
        
        sample = sample.float()
        return sample