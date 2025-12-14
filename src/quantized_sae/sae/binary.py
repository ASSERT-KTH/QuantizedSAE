import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class binary_decoder(nn.Module):
    def __init__(self, in_features, out_features, gamma=4.0, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.scale_factor = 2 ** n_bits
        self.gamma = gamma
        self.quantization_step = gamma / (2 ** (n_bits - 1))
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features*n_bits))
        self.bias = nn.Parameter(torch.zeros(out_features))

        nn.init.kaiming_normal_(self.weight)

    def forward(self, latent, true_sum):

        prob_weights = torch.sigmoid(self.weight)

        bit_weights_forward = (2 ** torch.arange(self.n_bits, device=prob_weights.device, dtype=prob_weights.dtype))
        bit_weights_forward[-1] *= -1

        # Reshape to [in_features, out_features, n_bits]
        prob_weights_reshaped = prob_weights.view(self.in_features, -1, self.n_bits)

        # Integer-valued effective weights using continuous bits (quantization-aware)
        int_weights = (prob_weights_reshaped * bit_weights_forward).sum(-1).float()  # [in_features, out_features]

        # Predicted reconstruction: [batch, out_features]
        reconstruction = self.quantization_step * latent.matmul(int_weights) + self.bias

        # sign_term = torch.where(prob_weights_reshaped < 0.5, 1.0, -1.0)
        # potential = 0.5 * prob_weights_reshaped.pow(2) - (1.0/3.0) * prob_weights_reshaped.pow(3)
        bit_weights_polarize = (2 ** torch.arange(self.n_bits, device=prob_weights.device, dtype=prob_weights.dtype))
        polarize_loss = (prob_weights_reshaped * (1 - prob_weights_reshaped) * bit_weights_polarize).mean()
        # polarize_contrib = potential * sign_term * bit_weights_polarize
        # polarize_loss = polarize_contrib.mean()

        return reconstruction, polarize_loss

    def quantized_int_weights(self):
        # Export quantized integer weights (two's complement) using 0.5 threshold
        with torch.no_grad():
            bits_hard = (torch.sigmoid(self.weight) > 0.5).to(self.weight.dtype)
            bits_hard = bits_hard.view(self.in_features, -1, self.n_bits)
            bit_weights_forward = (2 ** torch.arange(self.n_bits, device=bits_hard.device, dtype=bits_hard.dtype))
            bit_weights_forward = bit_weights_forward.clone()
            bit_weights_forward[-1] *= -1
            int_weights = (bits_hard * bit_weights_forward).sum(-1)
        return int_weights

    def quantized_int_weights_continuous(self):
        # Export quantized integer weights (two's complement) using 0.5 threshold
        with torch.no_grad():
            bits_hard = (torch.sigmoid(self.weight)).to(self.weight.dtype)
            bits_hard = bits_hard.view(self.in_features, -1, self.n_bits)
            bit_weights_forward = (2 ** torch.arange(self.n_bits, device=bits_hard.device, dtype=bits_hard.dtype))
            bit_weights_forward = bit_weights_forward.clone()
            bit_weights_forward[-1] *= -1
            int_weights = (bits_hard * bit_weights_forward).sum(-1)
        return int_weights

class BinarySAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, gamma=4.0, n_bits=8):

        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = 0.002

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        nn.init.xavier_uniform_(self.encoder[0].weight, gain=1)
        nn.init.zeros_(self.encoder[0].bias)

        self.decoder = binary_decoder(hidden_dim, input_dim, gamma=gamma, n_bits=self.n_bits)

    def forward(self, x):

        latent = self.encode(x)
        tok_values, tok_indices = latent.topk(int(self.hidden_dim * self.k), dim=1)
        # tok_values, tok_indices = latent.topk(32, dim=1)
        mask = torch.zeros_like(latent)
        mask.scatter_(1, tok_indices, 1.0)

        sparse_latent = latent * mask
        
        recon_loss, polarize_loss = self.decoder(sparse_latent, x)

        return sparse_latent, recon_loss, polarize_loss