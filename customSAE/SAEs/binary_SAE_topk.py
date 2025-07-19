import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder

class TopKBinaryDecoder(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.scale_factor = 2 ** n_bits - 1
        
        # Binary decoder weights
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features * n_bits))
        nn.init.uniform_(self.weight, -0.1, 0.1)

    def forward(self, latent, true_sum):
        # Get binary weights using straight-through estimator
        prob_weights = torch.sigmoid(self.weight)
        hard_bit = (prob_weights > 0.5).float()
        hard_weights = prob_weights + (hard_bit - prob_weights).detach()
        
        # Prepare latent (already binary from TopK)
        latent = latent.unsqueeze(-1)
        
        # Powers for signed integer representation
        powers = 2 ** torch.arange(self.n_bits, device=hard_weights.device).float()
        powers[-1] *= -1  # MSB is negative for signed
        
        # Convert to integers
        int_weights = (
            hard_weights.view(self.in_features, -1, self.n_bits) * powers
        ).sum(-1)
        int_sum = (
            true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
        ).sum(-1)
        
        # Compute prediction
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        
        # Loss
        loss = F.mse_loss(pred / self.scale_factor, int_sum / self.scale_factor)
        
        return loss


class TopKBinarySAE(SparseAutoencoder):
    """Binary SAE using TopK activation instead of sigmoid."""
    
    def __init__(self, input_dim, hidden_dim, n_bits=8, k=None):
        super().__init__(input_dim, hidden_dim)
        
        self.n_bits = n_bits
        self.k = k if k is not None else int(hidden_dim * 0.05)  # Default 5% sparsity
        
        # Encoder: Linear transformation without sigmoid
        self.encoder_linear = nn.Linear(input_dim * self.n_bits, hidden_dim, bias=True)
        
        # Better initialization
        nn.init.xavier_uniform_(self.encoder_linear.weight)
        nn.init.zeros_(self.encoder_linear.bias)
        
        # Decoder
        self.decoder = TopKBinaryDecoder(hidden_dim, input_dim, n_bits=self.n_bits)
    
    def topk_activation(self, x):
        """Apply TopK activation: only keep k largest values."""
        batch_size = x.shape[0]
        
        # Get top k values and indices
        topk_values, topk_indices = torch.topk(x, self.k, dim=-1)
        
        # Create binary mask
        mask = torch.zeros_like(x)
        mask.scatter_(-1, topk_indices, 1.0)
        
        # Straight-through estimator: use binary mask in forward, 
        # but gradients flow through original values
        return mask + (x - x.detach())
    
    def encode(self, x):
        # Linear transformation
        h = self.encoder_linear(x)
        
        # Apply TopK to get exactly k active neurons
        return self.topk_activation(h)
    
    def forward(self, x):
        # Encode with TopK
        latent = self.encode(x)
        
        # Decode and compute loss
        recon_loss = self.decoder(latent, x)
        
        return latent, recon_loss


class TopKBinarySAEWithSchedule(TopKBinarySAE):
    """Version with gradually decreasing k (annealing schedule)."""
    
    def __init__(self, input_dim, hidden_dim, n_bits=8, k_start=None, k_end=None):
        # Start with higher k
        k_start = k_start if k_start is not None else int(hidden_dim * 0.2)  # 20%
        super().__init__(input_dim, hidden_dim, n_bits, k=k_start)
        
        self.k_start = k_start
        self.k_end = k_end if k_end is not None else int(hidden_dim * 0.05)  # 5%
        self.current_step = 0
        self.anneal_steps = 10000  # Steps to go from k_start to k_end
    
    def update_k(self):
        """Gradually reduce k during training."""
        self.current_step += 1
        progress = min(self.current_step / self.anneal_steps, 1.0)
        self.k = int(self.k_start - progress * (self.k_start - self.k_end))
    
    def forward(self, x):
        # Update k according to schedule
        self.update_k()
        
        # Regular forward pass
        return super().forward(x)


if __name__ == "__main__":
    # Test the TopK Binary SAE
    print("Testing TopK Binary SAE...")
    
    batch_size = 32
    input_dim = 512
    hidden_dim = 16384
    n_bits = 8
    k = 820  # 5% of 16384
    
    model = TopKBinarySAE(input_dim, hidden_dim, n_bits, k)
    
    # Test input
    x = torch.randn(batch_size, input_dim * n_bits)
    x = (x > 0).float()  # Binary input
    
    # Forward pass
    latent, loss = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent shape: {latent.shape}")
    print(f"Active neurons per sample: {latent.sum(dim=-1)[0].item()} (exactly {k})")
    print(f"Loss: {loss.item():.4f}")
    
    # Test gradient flow
    loss.backward()
    
    enc_grad_norm = model.encoder_linear.weight.grad.norm().item()
    dec_grad_norm = model.decoder.weight.grad.norm().item()
    
    print(f"Encoder gradient norm: {enc_grad_norm:.4f}")
    print(f"Decoder gradient norm: {dec_grad_norm:.4f}") 