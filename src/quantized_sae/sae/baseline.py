import torch
import torch.nn as nn

class BaselineSparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        # Standard SAE Encoder: Linear -> ReLU
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim)
        )
        # Standard SAE Decoder: Linear (bias usually handled separately or included)
        self.decoder = nn.Linear(hidden_dim, input_dim)
        
        # Default K for analysis (optional)
        self.topk = 32

    def forward(self, x):
        """
        Args:
            x: Input tensor [batch, input_dim]
        """
        # 1. Encode (Linear + ReLU)
        h = self.encoder(x) 
        
        # 2. (Optional) Force Top-K Sparsity
        # This allows you to test "What if I only kept the top 32 features?"
        h = self.apply_topk_activation(h)

        # 3. Decode
        recon = self.decoder(h)
        
        return h, recon

    def apply_topk_activation(self, h):
        topk_values, topk_indices = torch.topk(h, self.topk, dim=1)
        
        # Create sparse tensor
        h_sparse = torch.zeros_like(h)
        h_sparse.scatter_(1, topk_indices, topk_values)
        return h_sparse
    
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