import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class QuantizedMatryoshkaDecoder(nn.Module):
    def __init__(self, in_features, out_features, abs_range=4, n_bits=8):

        super().__init__()
        self._ctx = [None] * n_bits
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.abs_range = abs_range
        self.quant_step = abs_range / (2 ** n_bits - 1)

        # Use torch.div with rounding_mode to avoid deprecated __rfloordiv__ warning
        # self.nested_dictionary_percentage = torch.div(
        #     torch.full((n_bits,), in_features, dtype=torch.long),
        #     2 ** torch.arange(n_bits, dtype=torch.long),
        #     rounding_mode='floor'
        # )

        self.nested_dictionary_size = ((in_features) * (2 ** torch.arange(n_bits, dtype=torch.long)) / 2 ** (n_bits-1)).int()

        # Learnable real-valued weights; STE binarized in forward
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
    
    def forward(self, latent):
        self._ctx = [None] * self.n_bits
        B = latent.size(0)
        reconstruction = torch.zeros(B, self.out_features, device=latent.device, dtype=latent.dtype)
        result = []
        start_idx = 0

        for i in range(self.n_bits):
            size_i = self.nested_dictionary_size[i].item() - start_idx
            if size_i == 0:
                result.append(reconstruction.clone())
                continue
                
            # STE binarization: forward uses sign, backward uses identity
            weight_slice = self.weight[start_idx:start_idx+size_i, :]

            Bsign = torch.where(
                weight_slice >= 0,
                torch.ones_like(weight_slice),
                -torch.ones_like(weight_slice),
            ).detach()

            ste_weight = (Bsign - weight_slice).detach() + weight_slice
            
            scale = 2 ** (self.n_bits - i - 1) * self.quant_step

            latent_slice = latent[:, start_idx:start_idx+size_i]
            reconstruction = reconstruction + scale * (latent_slice @ ste_weight)

            start_idx += size_i
            result.append(reconstruction.clone())

            self._ctx[i] = {
                'alpha': scale,
                'Bslice': Bsign,
                'z2': (latent_slice**2).sum(dim=0).detach(),
                'batch_size': B
            }

        return result

    @torch.no_grad()
    def apply_secant_grad(self):
        start = 0
        for i in range(self.n_bits):
            size = int(self.nested_dictionary_size[i].item()) - start
            if size == 0: 
                continue
            s, e = start, start + size
            alpha  = self._ctx[i]['alpha']
            z2     = self._ctx[i]['z2']
            Bslice = self._ctx[i]['Bslice']
            batch_size = self._ctx[i]['batch_size']

            c = 1.0 / batch_size / self.out_features
            m_i = self.n_bits - i

            # Base STE grad wrt W is already in self.weight.grad[s:e, :]
            # Convert base grad g  ->  secant grad g_sec = g - 2*alpha^2 * z2 * B
            self.weight.grad[s:e, :].add_(
                - 2 * c * m_i * (alpha**2) * z2[:, None] * Bslice
            )
            start = e

class QuantizedMatryoshkaSAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, top_k, abs_range=4, n_bits=8):
        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits
        self.abs_range = abs_range
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.top_k = top_k * 2 * n_bits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
        )

        nn.init.xavier_uniform_(self.encoder[0].weight, gain=1)
        nn.init.zeros_(self.encoder[0].bias)

        # Decoder using nested sub-dictionaries and STE-binarized weights
        self.decoder = QuantizedMatryoshkaDecoder(hidden_dim, input_dim, abs_range=abs_range, n_bits=n_bits)

    def forward(self, x):
        latent = self.encode(x)

        if self.top_k is not None and self.top_k > 0:
            k = min(self.top_k, latent.shape[-1])
            if k < latent.shape[-1]:
                _, topk_indices = torch.topk(latent.abs(), k, dim=-1)
                mask = torch.zeros_like(latent, dtype=torch.bool)
                mask.scatter_(dim=-1, index=topk_indices, value=True)
                latent = latent * mask

                latent = (latent.sign() - latent).detach() + latent

        return self.decode(latent)