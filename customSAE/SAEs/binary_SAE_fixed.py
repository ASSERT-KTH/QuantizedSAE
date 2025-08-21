import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder

class BinaryDecoderFixed(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8, gamma=2.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.gamma = gamma
        
        # For signed integers, the range is -2^(n-1) to 2^(n-1)-1
        # This should match the dataset's scale factor
        self.scale_factor = 2 ** (n_bits - 1) / (gamma + 1e-5)
        
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features * n_bits))
        nn.init.kaiming_normal_(self.weight)

    def forward(self, latent, true_sum):
        # Convert logits to hard bits using straight-through estimator
        prob_weights = torch.sigmoid(self.weight)
        hard_bit = (prob_weights > 0.5).float()
        hard_weights = prob_weights + (hard_bit - prob_weights).detach()
        
        latent = latent.unsqueeze(-1)
        
        # Powers for signed integer representation
        powers = 2 ** torch.arange(self.n_bits, device=hard_weights.device).float()
        powers[-1] *= -1  # MSB is negative for signed representation
        
        # Convert binary weights to signed integers
        int_weights = (
            hard_weights.view(self.in_features, -1, self.n_bits) * powers
        ).sum(-1)
        
        # Convert input binary representation to signed integers
        int_sum = (
            true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
        ).sum(-1)
        
        # Compute prediction
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        
        # Scale both prediction and target by the same factor before computing loss
        # This ensures the loss magnitude is reasonable
        pred_scaled = pred / self.scale_factor
        target_scaled = int_sum / self.scale_factor
        
        loss = F.mse_loss(pred_scaled, target_scaled)
        
        return loss


class BinarySAEFixed(SparseAutoencoder):
    def __init__(self, input_dim, hidden_dim, n_bits=8, gamma=2.0):
        super().__init__(input_dim, hidden_dim)
        
        self.n_bits = n_bits
        self.gamma = gamma
        
        # Encoder takes binary representation as input
        self.encoder = nn.Sequential(
            weight_norm(nn.Linear(input_dim * self.n_bits, hidden_dim), name="weight", dim=0),
            nn.Sigmoid()
        )
        
        # Better initialization for large hidden dimensions
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.zeros_(self.encoder[0].bias)
        
        self.decoder = BinaryDecoderFixed(hidden_dim, input_dim, n_bits=self.n_bits, gamma=gamma)
    
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        # Encode to get continuous latent activations
        latent = self.encode(x)
        
        # Convert to binary using straight-through estimator
        with torch.no_grad():
            binary_latent = (latent > 0.5).float()
        
        # Straight-through estimator: use binary values in forward pass but 
        # gradients flow through continuous values
        latent_ste = latent + (binary_latent - latent).detach()
        
        # Compute reconstruction loss
        loss = self.decoder(latent_ste, x)
        
        return latent_ste, loss 