import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder

class BinaryDecoderWithRegularization(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.scale_factor = 2 ** n_bits - 1
        
        # Decoder weights
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features * n_bits))
        nn.init.uniform_(self.weight, -0.1, 0.1)
    
    def binary_regularization_loss(self, reg_weight=1e-3):
        """Encourage weights to be close to 0 or 1 after sigmoid."""
        prob_weights = torch.sigmoid(self.weight)
        
        # Distance to nearest binary value
        dist_to_zero = torch.abs(prob_weights - 0.0)
        dist_to_one = torch.abs(prob_weights - 1.0)
        min_dist = torch.minimum(dist_to_zero, dist_to_one)
        
        # Regularization loss - penalize weights far from binary
        reg_loss = reg_weight * min_dist.mean()
        
        return reg_loss
    
    def forward(self, latent, true_sum, training=True, reg_weight=1e-3):
        # Use continuous weights during training
        prob_weights = torch.sigmoid(self.weight)
        
        # Add binary regularization during training
        reg_loss = 0
        if training:
            reg_loss = self.binary_regularization_loss(reg_weight)
        
        # Use continuous weights for computation
        hard_weights = prob_weights
        
        # Rest of forward pass
        latent = latent.unsqueeze(-1)
        powers = 2 ** torch.arange(self.n_bits, device=hard_weights.device).float()
        powers[-1] *= -1
        
        int_weights = (
            hard_weights.view(self.in_features, -1, self.n_bits) * powers
        ).sum(-1)
        int_sum = (
            true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
        ).sum(-1)
        
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        recon_loss = F.mse_loss(pred / self.scale_factor, int_sum / self.scale_factor)
        
        return recon_loss + reg_loss, recon_loss, reg_loss


class BinarySAEImprovedPostQuant(SparseAutoencoder):
    def __init__(self, input_dim, hidden_dim, n_bits=8):
        super().__init__(input_dim, hidden_dim)
        
        self.n_bits = n_bits
        
        # Encoder: Convert binary to int, then encode
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Better initialization
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.constant_(self.encoder[0].bias, -2.0)  # Encourage sparsity
        
        # Decoder with regularization
        self.decoder = BinaryDecoderWithRegularization(hidden_dim, input_dim, n_bits=self.n_bits)
    
    def bin2int(self, x):
        """Convert binary representation to integers."""
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.input_dim, self.n_bits)
        
        powers = 2 ** torch.arange(self.n_bits, device=x.device).float()
        powers[-1] *= -1  # Signed representation
        
        integers = (x_reshaped * powers).sum(-1)
        return integers
    
    def forward(self, x, reg_weight=1e-3):
        # Convert binary input to integers
        x_int = self.bin2int(x)
        
        # Encode
        latent_continuous = self.encoder(x_int)
        
        # Apply STE for binary latents
        with torch.no_grad():
            binary_latent = (latent_continuous > 0.5).float()
        latent = latent_continuous + (binary_latent - latent_continuous).detach()
        
        # Decode with regularization
        total_loss, recon_loss, reg_loss = self.decoder(
            latent, x, training=self.training, reg_weight=reg_weight
        )
        
        return latent, total_loss, recon_loss, reg_loss
    
    def quantize_decoder_weights(self):
        """Quantize decoder weights for inference."""
        with torch.no_grad():
            prob_weights = torch.sigmoid(self.decoder.weight)
            binary_weights = (prob_weights > 0.5).float()
            
            # Convert back to logits
            binary_logits = torch.log(binary_weights.clamp(1e-7, 1-1e-7) / 
                                    (1 - binary_weights.clamp(1e-7, 1-1e-7)))
            
            self.decoder.weight.data = binary_logits


class BinarySAEWithGumbelSoftmax(SparseAutoencoder):
    """Alternative: Use Gumbel-Softmax for differentiable discrete weights."""
    
    def __init__(self, input_dim, hidden_dim, n_bits=8):
        super().__init__(input_dim, hidden_dim)
        
        self.n_bits = n_bits
        self.temperature = 1.0  # For Gumbel-Softmax
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        nn.init.xavier_uniform_(self.encoder[0].weight)
        nn.init.constant_(self.encoder[0].bias, -2.0)
        
        # Decoder weights as logits for binary Gumbel-Softmax
        self.decoder_weight_logits = nn.Parameter(
            torch.randn(hidden_dim, input_dim * n_bits, 2) * 0.1
        )
    
    def gumbel_sigmoid(self, logits, temperature=1.0):
        """Gumbel-Softmax for binary variables."""
        # Sample Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        
        # Apply Gumbel-Softmax
        y = torch.softmax((logits + gumbel_noise) / temperature, dim=-1)
        
        # Take the probability of being 1 (second class)
        return y[..., 1]
    
    def bin2int(self, x):
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.input_dim, self.n_bits)
        
        powers = 2 ** torch.arange(self.n_bits, device=x.device).float()
        powers[-1] *= -1
        
        integers = (x_reshaped * powers).sum(-1)
        return integers
    
    def forward(self, x):
        # Convert binary to integers
        x_int = self.bin2int(x)
        
        # Encode
        latent_continuous = self.encoder(x_int)
        
        # STE for encoder
        with torch.no_grad():
            binary_latent = (latent_continuous > 0.5).float()
        latent = latent_continuous + (binary_latent - latent_continuous).detach()
        
        # Get decoder weights using Gumbel-Softmax
        if self.training:
            decoder_weights = self.gumbel_sigmoid(
                self.decoder_weight_logits, self.temperature
            )
        else:
            # Use hard weights during inference
            decoder_weights = torch.softmax(self.decoder_weight_logits, dim=-1)[..., 1]
        
        # Decode
        latent_expanded = latent.unsqueeze(-1)
        powers = 2 ** torch.arange(self.n_bits, device=x.device).float()
        powers[-1] *= -1
        
        int_weights = (
            decoder_weights.view(latent.shape[1], -1, self.n_bits) * powers
        ).sum(-1)
        int_sum = (
            x.view(x.shape[0], -1, self.n_bits) * powers
        ).sum(-1)
        
        pred = (latent_expanded * int_weights.unsqueeze(0)).sum(-2)
        loss = F.mse_loss(pred / 255, int_sum / 255)
        
        return latent, loss


if __name__ == "__main__":
    # Test the improved approaches
    print("Testing Improved Post-Quantization Binary SAE...")
    
    batch_size = 32
    input_dim = 512
    hidden_dim = 2048
    n_bits = 8
    
    # Test regularized approach
    model1 = BinarySAEImprovedPostQuant(input_dim, hidden_dim, n_bits)
    x = torch.randn(batch_size, input_dim * n_bits)
    x = (x > 0).float()
    
    latent, total_loss, recon_loss, reg_loss = model1(x, reg_weight=1e-3)
    
    print(f"Regularized approach:")
    print(f"  Total loss: {total_loss.item():.4f}")
    print(f"  Recon loss: {recon_loss.item():.4f}")
    print(f"  Reg loss: {reg_loss.item():.4f}")
    print(f"  Activated neurons: {latent.sum(dim=-1).mean():.1f}")
    
    # Test Gumbel-Softmax approach
    print(f"\nTesting Gumbel-Softmax approach...")
    model2 = BinarySAEWithGumbelSoftmax(input_dim, hidden_dim, n_bits)
    latent, loss = model2(x)
    
    print(f"Gumbel-Softmax approach:")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Activated neurons: {latent.sum(dim=-1).mean():.1f}") 