import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm
import os
import sys 
sys.path.insert(0, os.getcwd())
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class binary_decoder(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.scale_factor = 2 ** (n_bits - 1)
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features*n_bits))

        nn.init.kaiming_normal_(self.weight)

    def store_pre_update_state(self):

        with torch.no_grad():
            prob_weights = torch.sigmoid(self.weight)
            hard_bit = (prob_weights > 0.5).float()
            self.prev_weight_states = hard_bit

    def check_threshold_crossings(self):

        if self.prev_weight_states is None:
            return
        
        with torch.no_grad():
            current_states = (torch.sigmoid(self.weight) > 0.5)
            crossings = (self.prev_weight_states != current_states)
            
            if not crossings.any():
                return None, None
            
            # Reshape crossings to [in_features, out_features, n_bits]
            current_states_reshaped = current_states.view(self.in_features, -1, self.n_bits)
            crossings_reshaped = crossings.view(self.in_features, -1, self.n_bits)
            
            max_indices = torch.full((self.in_features, crossings_reshaped.shape[1]), -1, 
                                   dtype=torch.long, device=crossings.device)
            
            first_flip_value = torch.full((self.in_features, crossings_reshaped.shape[1]), -1, 
                                   dtype=torch.int8, device=crossings.device)

            # For each atom, find the highest index (MSB) that has a crossing
            for i in range(self.n_bits - 1, -1, -1):  # From MSB to LSB
                atom_has_crossing_at_bit = crossings_reshaped[:, :, i]
                bit_in_pos = current_states_reshaped[:, :, i]
                max_indices = torch.where((max_indices == -1) & atom_has_crossing_at_bit, 
                                        i, max_indices)
                first_flip_value = torch.where((first_flip_value == -1) & atom_has_crossing_at_bit, 
                                        bit_in_pos, first_flip_value)
            
            for i in range(self.n_bits):
                self.weight.data[:, i::self.n_bits] = torch.where((i < max_indices) & (max_indices != (self.n_bits - 1)), 
                                                                 0. - (first_flip_value - 0.5) / 500., self.weight.data[:, i::self.n_bits])
                self.weight.data[:, i::self.n_bits] = torch.where((i < max_indices) & (max_indices == (self.n_bits - 1)), 
                                                                 0. + (first_flip_value - 0.5) / 500., self.weight.data[:, i::self.n_bits])

    def forward(self, latent, true_sum):

        prob_weights = torch.sigmoid(self.weight)
        hard_bit = (prob_weights > 0.5).float()
        # hard_weights = prob_weights + (hard_bit - prob_weights).detach()
        hard_weights = prob_weights
        
        latent = latent.unsqueeze(-1)
        powers = 2 ** torch.arange(self.n_bits, device=hard_weights.device)
        powers[-1] *= -1
        int_weights = (
            hard_weights.view(self.in_features, -1, self.n_bits) * powers
        ).sum(-1).float()
        int_sum = (
            true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
        ).sum(-1).float()
        
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        polarize_loss = (prob_weights*(1-prob_weights)).mean()
        recon_loss = 0.5 * ((pred - int_sum).float().pow(2)/self.scale_factor).mean()

        return recon_loss, polarize_loss

class TemperatureSigmoid(nn.Module):
    def __init__(self, temperature=1.0):
        super().__init__()
        self.temperature = temperature
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        return self.sigmoid(x / self.temperature)

class BinarySAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, n_bits=8):

        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.k = 0.002

        self.encoder = nn.Sequential(
            # weight_norm(nn.Linear(input_dim*self.n_bits, hidden_dim), name="weight", dim=0),
            nn.Linear(input_dim*self.n_bits, hidden_dim),
            # nn.Linear(input_dim, hidden_dim)
            # nn.LayerNorm(hidden_dim, eps=1e-05),
            # TemperatureSigmoid(temperature=0.5)
            # nn.Sigmoid()
        )

        # nn.init.normal_(self.encoder[0].weight, std=torch.sqrt(torch.tensor(2/hidden_dim)))
        # nn.init.normal_(self.encoder[0].weight, std=torch.sqrt(torch.tensor(1/hidden_dim)))
        nn.init.xavier_uniform_(self.encoder[0].weight, gain=1)
        nn.init.zeros_(self.encoder[0].bias)

        self.decoder = binary_decoder(hidden_dim, input_dim, n_bits=self.n_bits)

    def store_decoder_pre_update_state(self):
        self.decoder.store_pre_update_state()

    def check_decoder_threshold_crossings(self):
        return self.decoder.check_threshold_crossings()
    
    def bin2int(self, x):
        # x shape: [batch, input_dim*n_bits]
        # Reshape to [batch, input_dim, n_bits]
        batch_size = x.shape[0]
        x_reshaped = x.view(batch_size, self.input_dim, self.n_bits)
        
        # Create powers of 2 for binary to integer conversion
        powers = 2 ** torch.arange(self.n_bits, device=x.device)
        powers[-1] *= -1  # Make MSB negative for signed representation
        
        # Convert binary to integer: multiply by powers and sum over n_bits dimension
        integers = (x_reshaped * powers).sum(-1)
        
        return integers  # shape: [batch, input_dim]

    def forward(self, x):

        # x_int = self.bin2int(x)
        # latent = self.encode(x_int)
        latent = self.encode(x)
        th = latent.topk(int(self.hidden_dim * self.k), dim=1).values[:, -1:]
        binary = (latent >= th).float()
        latent = latent + binary - latent.detach()

        with torch.no_grad():
            binary_latent = (latent > 0).float()

        latent = latent + binary_latent - latent.detach()
        
        recon_loss, polarize_loss = self.decoder(latent, x)

        return latent, recon_loss, polarize_loss

# # Setup
# model = BinarySAE(2, 2, 4)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# 
# num_epochs = 1000
# # Training loop with threshold crossing detection
# for epoch in range(num_epochs):
#     model.store_decoder_pre_update_state()
#     
#     x = torch.randn(1, 8) > 0.5
#     latent, loss, trigger = model(x.float())
#     
#     optimizer.zero_grad()
#     trigger.backward()
#     loss.backward()
#     optimizer.step()
#     
#     # Check for threshold crossings after update
#     model.check_decoder_threshold_crossings()