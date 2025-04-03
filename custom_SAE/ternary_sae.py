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

        self.register_buffer('mask', torch.ones_like(self.weight))
        self.register_buffer('input_activations', None)
        self.register_buffer('output_grad', None)

        self.register_forward_hook(self._capture_input_activations)
        self.register_backward_hook(self._capture_output_grad)

        nn.init.kaiming_normal_(self.weight)
        
    def _capture_input_activations(self, module, inp, out):
        self.input_activations = inp[0].detach()

    def _capture_output_grad(self, module, grad_input, grad_output):
        self.output_grad = grad_output[0].detach()

    def init_mask(self, sparsity):
        with torch.no_grad():
            num_weights = self.weight.numel()
            num_inactive = int(num_weights * sparsity)
            
            scores = self.weight.abs().flatten()
            _, topk_indices = torch.topk(scores, num_inactive, largest=False)
            
            initial_mask = torch.ones_like(scores)
            initial_mask[topk_indices] = 0
            initial_mask = initial_mask.view_as(self.weight)
            self.mask.data = initial_mask
            self.weight.data *= self.mask.data

    def forward(self, x):
        # FP:
        # return F.linear(x, self.weight)

        # Ternary:
        with torch.no_grad():
            sign_weight = torch.sign(self.weight)
            mask = (torch.abs(self.weight) >= self.threshold).float()
            hard_weights = sign_weight * mask # Ternary SAE
        # hard_weights = torch.sign(self.weight)  # Binary
        masked_weight = self.weight * self.mask
        return F.linear(x, masked_weight + (hard_weights - masked_weight).detach())

    def update_mask(self, f_decay, sparsity_rate=0.7):
        with torch.no_grad():
            flat_weights = self.weight.data.flatten()
            flat_mask = self.mask.flatten()
            active = flat_mask.bool()
            
            n_drop = int(f_decay * (1 - sparsity_rate) * flat_weights.size(0))
            n_grow = n_drop
            
            # 1. Drop: Prune smallest active weights
            if n_drop > 0:
                active_weights = flat_weights[active].abs()
                threshold = torch.kthvalue(active_weights, k=n_drop)[0]
                drop_mask = (flat_weights.abs() <= threshold) & active
                active[drop_mask] = False

            # 2. Grow: Select using gradient magnitude
            if n_grow > 0 and self.input_activations is not None:
                # Calculate gradient scores using captured data
                a = self.input_activations.mean(dim=0)  # (in_features)
                δ = self.output_grad.mean(dim=0)        # (out_features)
                scores = torch.outer(δ.abs(), a.abs()).flatten()
                
                # Only consider inactive locations
                scores[active] = -float('inf')
                
                # Select top-k inactive locations
                _, grow_indices = torch.topk(scores, n_grow)
                active[grow_indices] = True
                
            # Update mask and restore original shape
            self.mask.data = active.view_as(self.mask).float()
            self.weight.data *= self.mask.data
    
    def mask_grad(self):
        self.weight.grad.data *= self.mask.data

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