import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder

class BinaryDecoderV2(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.scale_factor = 2 ** n_bits - 1
        
        # Initialize decoder weights
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features * n_bits))
        
        # Better initialization for binary weights
        # Initialize around 0 so sigmoid(weight) ≈ 0.5
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, latent, true_sum):
        # Get binary weights using straight-through estimator
        prob_weights = torch.sigmoid(self.weight)
        hard_bit = (prob_weights > 0.5).float()
        hard_weights = prob_weights + (hard_bit - prob_weights).detach()
        
        # Prepare latent
        latent = latent.unsqueeze(-1)
        
        # Powers for signed integer representation
        powers = 2 ** torch.arange(self.n_bits, device=hard_weights.device).float()
        powers[-1] *= -1  # MSB is negative for signed
        
        # Convert weights and input to integers
        int_weights = (
            hard_weights.view(self.in_features, -1, self.n_bits) * powers
        ).sum(-1)
        int_sum = (
            true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
        ).sum(-1)
        
        # Compute prediction
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        
        # MSE loss with normalization
        loss = F.mse_loss(pred / self.scale_factor, int_sum / self.scale_factor)
        
        return loss


class BinarySAEV2(SparseAutoencoder):
    def __init__(self, input_dim, hidden_dim, n_bits=8, use_bias=True):
        super().__init__(input_dim, hidden_dim)
        
        self.n_bits = n_bits
        self.use_bias = use_bias
        
        # Simple linear encoder without weight norm
        self.encoder_linear = nn.Linear(input_dim * self.n_bits, hidden_dim, bias=use_bias)
        
        # Better initialization for large hidden dimensions
        # Use Xavier/Glorot initialization
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        if use_bias:
            # Initialize bias to slightly negative to encourage sparsity
            nn.init.constant_(self.encoder_linear.bias, -1.0)
        
        # Decoder
        self.decoder = BinaryDecoderV2(hidden_dim, input_dim, n_bits=self.n_bits)
        
    def encode(self, x):
        # Linear transformation followed by sigmoid
        h = self.encoder_linear(x)
        return torch.sigmoid(h)
    
    def forward(self, x):
        # Encode
        latent_continuous = self.encode(x)
        
        # Binarize using straight-through estimator
        latent_binary = (latent_continuous > 0.5).float()
        latent = latent_continuous + (latent_binary - latent_continuous).detach()
        
        # Decode and compute loss
        recon_loss = self.decoder(latent, x)
        
        return latent, recon_loss


class BinarySAEV2WithSparsity(BinarySAEV2):
    """Version with built-in sparsity loss."""
    
    def forward(self, x, sparsity_weight=1e-5):
        # Get latent and reconstruction loss
        latent, recon_loss = super().forward(x)
        
        # Add L1 sparsity penalty on the continuous latent
        # This encourages the sigmoid to output values close to 0
        latent_continuous = self.encode(x)
        sparsity_loss = sparsity_weight * latent_continuous.mean()
        
        total_loss = recon_loss + sparsity_loss
        
        return latent, total_loss, recon_loss, sparsity_loss


if __name__ == "__main__":
    # Quick test
    model = BinarySAEV2(input_dim=512, hidden_dim=16384, n_bits=8)
    
    # Test input
    batch_size = 32
    x = torch.randn(batch_size, 512 * 8)
    x = (x > 0).float()  # Binary input
    
    # Forward pass
    latent, loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Loss: {loss.item():.4f}")
    print(f"Activated neurons: {latent.sum(dim=-1).mean():.1f}") 