import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder
from SAEs.quantized_matryoshka_SAE import *
from nnba.adder import *

class ResidualQuantizedSAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, top_k, abs_range=4, n_bits=8):
        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits
        self.abs_range = abs_range
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.top_k = top_k

        # Calculate the size of each nested dictionary level
        # Pattern: [1, 1, 2, 4, 8, ..., 2^(n_bits-2)] / 2^(n_bits-1)
        sizes = []
        for i in range(n_bits):
            if i < 2:
                sizes.append(1)
            else:
                sizes.append(2 ** (i - 1))
        
        current_sum = sum(sizes)
        
        if current_sum != hidden_dim:
            scale_factor = hidden_dim / current_sum
            sizes = [max(1, int(s * scale_factor)) for s in sizes]
            sizes[-1] = hidden_dim - sum(sizes[:-1])
        
        self.sae_hidden_dims = sizes

        self.saes = nn.ModuleList()
        for i in range(self.n_bits):
            # abs_range = abs_range / 2 ** i
            sae = QuantizedMatryoshkaSAE(
                input_dim=input_dim,
                hidden_dim=self.sae_hidden_dims[i],
                top_k=top_k,
                abs_range=abs_range,
                n_bits=1,
                allow_bias=True if i == 0 else False
            )
            self.saes.append(sae)

    def forward(self, x):

        residual = x
        all_latent_groups = []
        all_reconstruction_levels = []
        
        for sae in self.saes:

            latent_group, reconstructions = sae(residual)
            all_latent_groups.append(latent_group[-1])
            all_reconstruction_levels.append(reconstructions[-1])
            
            reconstruction = reconstructions[-1]
            
            residual = (residual - reconstruction).detach() * 2
        
        return all_latent_groups, all_reconstruction_levels

    def apply_secant_grad(self):

        for sae in self.saes:
            sae.decoder.apply_secant_grad()