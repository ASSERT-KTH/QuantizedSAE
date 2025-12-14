import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class QuantizedMatryoshkaDecoder(nn.Module):
    def __init__(self, in_features, out_features, abs_range=4, n_bits=8, top_k=None, joint_gradient=False, allow_bias=True):

        super().__init__()
        self._ctx = [None] * n_bits
        self.joint_gradient = joint_gradient
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.abs_range = abs_range
        self.quant_step = abs_range / (2 ** (n_bits - 1))
        # self.quant_step = abs_range / (2 ** n_bits)
        self.top_k = top_k
        self.allow_bias = allow_bias

        # self.nested_dictionary_size = ((in_features) * (2 ** torch.arange(n_bits, dtype=torch.long)) / 2 ** (n_bits-1)).int()
        self.nested_dictionary_size = []
        for i in range(n_bits):
            if i < 2:
                self.nested_dictionary_size.append(1)
            else:
                self.nested_dictionary_size.append(2 ** (i - 1))
        
        current_sum = sum(self.nested_dictionary_size)
        
        if current_sum != in_features:
            scale_factor = in_features / current_sum
            self.nested_dictionary_size = [max(1, int(s * scale_factor)) for s in self.nested_dictionary_size]
            self.nested_dictionary_size[-1] = in_features - sum(self.nested_dictionary_size[:-1])

        # Learnable real-valued weights; STE binarized in forward
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.weight_mirror = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.weight_mirror)
    
    def forward(self, latent):
        self._ctx = [None] * self.n_bits
        B = latent.size(0)
        reconstruction = torch.zeros(B, self.out_features, device=latent.device, dtype=latent.dtype)
        latent_group = []
        result = []
        start_idx = 0

        for i in range(self.n_bits):
            # size_i = self.nested_dictionary_size[i] - start_idx
            size_i = self.nested_dictionary_size[i]
            if size_i == 0:
                result.append(reconstruction.clone())
                continue
                
            # STE binarization: forward uses sigmoid + threshold, backward uses identity
            weight_slice = self.weight[start_idx:start_idx+size_i, :]
            weight_slice_mirror = self.weight_mirror[start_idx:start_idx+size_i, :]

            # Apply sigmoid and threshold at 0.5 to get binary {0, 1}
            weight_sigmoid = torch.sigmoid(weight_slice)
            weight_sigmoid_mirror = torch.sigmoid(weight_slice_mirror)
            
            Bsign = torch.where(
                weight_sigmoid >= 0.5,
                torch.ones_like(weight_sigmoid),
                -torch.ones_like(weight_sigmoid),
            ).detach()

            Bsign_mirror = torch.where(
                weight_sigmoid_mirror >= 0.5,
                torch.ones_like(weight_sigmoid_mirror),
                -torch.ones_like(weight_sigmoid_mirror),
            ).detach()

            l2_norms_of_Bsign = torch.norm(Bsign + Bsign_mirror, p=2, dim=1)
            
            # Check for potential numerical issues
            if (l2_norms_of_Bsign < 1e-6).any():
                print(f"Warning: Very small norm detected at level {i}: min={l2_norms_of_Bsign.min().item():.2e}")

            scale_factor = 2 ** (self.n_bits - i - 2) * self.quant_step

            scale_vector = scale_factor / (l2_norms_of_Bsign + 1e-8)
            # scale_vector = (scale_factor / torch.ones_like(l2_norms_of_Bsign)).detach()

            # STE: forward uses binary, backward flows through sigmoid
            ste_weight = (Bsign - weight_sigmoid).detach() + weight_sigmoid
            ste_weight_mirror = (Bsign_mirror - weight_sigmoid_mirror).detach() + weight_sigmoid_mirror

            latent_slice = latent[:, start_idx:start_idx+size_i]

            latent_slice = ((latent_slice > 0.5).to(latent_slice.dtype) - latent_slice).detach() + latent_slice

            # Precompute hook constants detached to avoid capturing large tensors/graphs
            with torch.no_grad():
                wsum = (Bsign + Bsign_mirror)
                l2_norms_of_Wsum = torch.norm(wsum, p=2, dim=1)
                wnorm_sq = ((scale_vector * l2_norms_of_Wsum) ** 2).unsqueeze(0)  # [1, size_i]

            z_bin_detached = (latent_slice.detach() > 0).to(latent_slice.dtype)
            z_bias = (z_bin_detached - 0.5).detach()  # [B, size_i]
            c_const = (1.0 / B / self.out_features)

            def latent_hook(g, z_bias=z_bias, wnorm_sq=wnorm_sq, c=c_const):
                if self.joint_gradient:
                    return g - (self.n_bits - i) * c * z_bias * wnorm_sq
                else:
                    return g - c * z_bias * wnorm_sq

            # latent_slice.register_hook(latent_hook)

            # reconstruction = reconstruction.detach() + scale * (latent_slice @ (ste_weight + ste_weight_mirror))
            reconstruction = reconstruction if self.joint_gradient else reconstruction.detach()
            reconstruction += (scale_vector * latent_slice) @ (ste_weight + ste_weight_mirror)

            if i == 0 and self.allow_bias:
                reconstruction = reconstruction + self.bias

            start_idx += size_i
            # latent_group.append(latent_slice.abs().sum(dim=-1).mean().clone())
            latent_group.append(latent_slice.sum(dim=-1).mean().clone())
            result.append(reconstruction.clone())

            self._ctx[i] = {
                # 'alpha': scale,
                'alpha_vector': scale_vector,
                'Bslice': Bsign.to(torch.int8),
                'Bslice_mirror': Bsign_mirror.to(torch.int8),
                # 'z2': (latent_slice**2).sum(dim=0).detach(),
                'z2': latent_slice.sum(dim=0).detach(), # Since latent_slice is binary, z2 is the sum of latent_slice
                'batch_size': B,
                'weight_sigmoid': weight_sigmoid.detach(),
                'weight_sigmoid_mirror': weight_sigmoid_mirror.detach()
            }

        return latent_group, result

    @torch.no_grad()
    def apply_secant_grad(self):
        start = 0
        for i in range(self.n_bits):
            # size = int(self.nested_dictionary_size[i]) - start
            size = int(self.nested_dictionary_size[i])
            if size == 0: 
                continue
            s, e = start, start + size
            # alpha  = self._ctx[i]['alpha']
            alpha_vector = self._ctx[i]['alpha_vector']
            z2     = self._ctx[i]['z2']
            # Ensure float dtype for arithmetic
            Bslice = self._ctx[i]['Bslice'].to(alpha_vector.dtype)
            Bslice_mirror = self._ctx[i]['Bslice_mirror'].to(alpha_vector.dtype)
            batch_size = self._ctx[i]['batch_size']
            weight_sigmoid = self._ctx[i]['weight_sigmoid']
            weight_sigmoid_mirror = self._ctx[i]['weight_sigmoid_mirror']

            c = 1.0 / batch_size / self.out_features
            m_i = self.n_bits - i

            # Compute sigmoid derivative: sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            sigmoid_grad = weight_sigmoid * (1.0 - weight_sigmoid)
            sigmoid_grad_mirror = weight_sigmoid_mirror * (1.0 - weight_sigmoid_mirror)

            # Base STE grad wrt W is already in self.weight.grad[s:e, :]
            # Convert base grad g  ->  secant grad g_sec = g - c * z2 * alpha^2 * B
            # Apply chain rule: multiply by sigmoid derivative
            if self.joint_gradient:
                self.weight.grad[s:e, :].add_(
                    - c * m_i * (z2 * alpha_vector**2)[:, None] * Bslice * sigmoid_grad
                )

                self.weight_mirror.grad[s:e, :].add_(
                    - c * m_i * (z2 * alpha_vector**2)[:, None] * Bslice_mirror * sigmoid_grad_mirror
                )
            else:
                self.weight.grad[s:e, :].add_(
                    - c * (z2 * alpha_vector**2)[:, None] * Bslice * sigmoid_grad
                )

                self.weight_mirror.grad[s:e, :].add_(
                    - c * (z2 * alpha_vector**2)[:, None] * Bslice_mirror * sigmoid_grad_mirror
                )
            start = e

class QuantizedMatryoshkaSAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, top_k, abs_range=4, n_bits=8, allow_bias=True):
        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits
        self.abs_range = abs_range
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.allow_bias = allow_bias

        # self.top_k = top_k * 2 * n_bits
        self.top_k = top_k

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        nn.init.xavier_uniform_(self.encoder[0].weight, gain=1)
        nn.init.zeros_(self.encoder[0].bias)

        # Decoder using nested sub-dictionaries and STE-binarized weights
        self.decoder = QuantizedMatryoshkaDecoder(hidden_dim, input_dim, abs_range=abs_range, n_bits=n_bits, top_k=self.top_k, allow_bias=self.allow_bias)

    def forward(self, x):
        latent = self.encode(x)

        return self.decode(latent)