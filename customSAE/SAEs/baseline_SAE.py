import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class BaselineSparseAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)
        self.topk = int(hidden_dim * 0.002)
        
    def normalize_decoder_weights(self):
        """Normalize decoder weights to unit norm along the input dimension"""
        with torch.no_grad():
            # Normalize each column (decoder neuron) to unit norm
            weight = self.decoder.weight.data  # Shape: (input_dim, hidden_dim)
            # Normalize along dimension 0 (input_dim) to get unit norm for each hidden unit
            norms = torch.norm(weight, dim=0, keepdim=True)
            # Avoid division by zero
            norms = torch.clamp(norms, min=1e-8)
            self.decoder.weight.data = weight / norms
        
    def apply_topk_activation(self, h):

        batch_size, hidden_dim = h.shape
        
        # Get top-k indices for each sample in the batch
        topk_values, topk_indices = torch.topk(h, self.topk, dim=1)
        
        # Create sparse activation tensor
        topk_values = torch.where(topk_values > 0, topk_values, torch.zeros_like(topk_values))
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(1, topk_indices, topk_values)
        
        return h_sparse
        
    def forward(self, x):
        h = self.encoder(x)
        
        # Apply top-k activation for sparsity
        # h_sparse = self.apply_topk_activation(h)

        return h, self.decoder(h)