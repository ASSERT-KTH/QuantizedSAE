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
    def __init__(self, in_features, out_features, n_bits=8, polarize_factor=0.01):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features*n_bits))
        self.threshold = 0.5 # For binary SAE
        self.polarize_factor = polarize_factor

        # Threshold crossing tracking
        self.prev_weight_states = None
        self.threshold_crossing_callback = None

        # self.csa = carry_save_adder(n_bits)
        # self.sga = surrogate_gradient_adder_dense(n_bits)
        self.ra = residual_adder(n_bits)

        # for param in self.csa.parameters():
        #     param.requires_grad = False

        # nn.init.kaiming_normal_(self.weight)
        # nn.init.normal_(self.weight, mean=0.5, std=0.2)
        nn.init.normal_(self.weight, mean=0.5, std=0.0001)
        # Clamp outliers to [0, 1]
        self.weight.data.clamp_(0, 1)
        
        self.hook_handle = None
        # self.register_hook()

    def store_pre_update_state(self):
        with torch.no_grad():
            self.prev_weight_states = (self.weight >= self.threshold).clone()

    def check_threshold_crossings(self):

        if self.prev_weight_states is None:
            return
        
        with torch.no_grad():
            current_states = (self.weight >= self.threshold)
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
                                                                 0.5 - (first_flip_value - 0.5) / 50., self.weight.data[:, i::self.n_bits])
                self.weight.data[:, i::self.n_bits] = torch.where((i < max_indices) & (max_indices == (self.n_bits - 1)), 
                                                                 0.5 + (first_flip_value - 0.5) / 50., self.weight.data[:, i::self.n_bits])

    def register_hook(self):

        def weight_hook(grad):
            distance_from_threshold = torch.where(grad > 0, self.weight, 1 - self.weight)
            # distance_from_threshold = self.weight * (1 - self.weight)

            return grad * distance_from_threshold

        self.hook_handle = self.weight.register_hook(weight_hook)

    def forward(self, latent, true_sum):
        # Binary weights(feature representations):
        # self.weight.data.clamp_(0, 1) # Problematic part
        # clamped_weight = self.weight.clamp(0, 1)
        hard_weights = self.weight + ((self.weight > self.threshold).float() - self.weight).detach() # Binary SAE x = x.unsqueeze(-1)
        
        latent = latent.unsqueeze(-1)
        with torch.no_grad():
            hard_weights_copy = hard_weights.clone()
            powers = 2 ** torch.arange(self.n_bits, device=hard_weights_copy.device)
            powers[-1] *= -1
            int_weights = (
                hard_weights_copy.view(self.in_features, -1, self.n_bits) * powers
            ).sum(-1).float()
            int_sum = (
                true_sum.view(true_sum.shape[0], -1, self.n_bits) * powers
            ).sum(-1).float()
        
        pred = (latent * int_weights.unsqueeze(0)).sum(-2)
        loss = 0.5 * (pred - int_sum).float().pow(2).mean()

        # For training the decoder:
        with torch.no_grad():
            latent_copy = latent.clone()

        filtered_features = (latent_copy * hard_weights.unsqueeze(0))

        # Batch, Feature, Neuron, N_bits
        features_by_neurons = filtered_features.unfold(-1, self.n_bits, self.n_bits).permute(0, 2, 1, 3)
        features_by_neurons = features_by_neurons.reshape(-1, features_by_neurons.shape[-2], features_by_neurons.shape[-1])

        true_sum_by_neurons = true_sum.repeat(1, self.in_features, 1)
        true_sum_by_neurons = true_sum_by_neurons.unfold(-1, self.n_bits, self.n_bits).permute(0, 2, 1, 3)
        true_sum_by_neurons = true_sum_by_neurons.reshape(-1, true_sum_by_neurons.shape[-2], true_sum_by_neurons.shape[-1])

        weight_repeat = self.weight.repeat(latent.shape[0], 1, 1)
        weight_repeat_by_neurons = weight_repeat.unfold(-1, self.n_bits, self.n_bits).permute(0, 2, 1, 3)
        weight_repeat_by_neurons = weight_repeat_by_neurons.reshape(-1, weight_repeat_by_neurons.shape[-2], weight_repeat_by_neurons.shape[-1])

        trigger = self.ra(features_by_neurons, weight_repeat_by_neurons, true_sum_by_neurons)

        return loss, trigger

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

        self.encoder = nn.Sequential(
            weight_norm(nn.Linear(input_dim*self.n_bits, hidden_dim), name="weight", dim=0),
            nn.LayerNorm(hidden_dim, eps=1e-05),
            TemperatureSigmoid(temperature=0.5)
        )

        # nn.init.normal_(self.encoder[0].weight, std=torch.sqrt(torch.tensor(2/hidden_dim)))
        nn.init.normal_(self.encoder[0].weight, std=torch.sqrt(torch.tensor(1/hidden_dim)))
        nn.init.zeros_(self.encoder[0].bias)

        self.decoder = binary_decoder(hidden_dim, input_dim, n_bits=self.n_bits)

    def store_decoder_pre_update_state(self):
        self.decoder.store_pre_update_state()

    def check_decoder_threshold_crossings(self):
        return self.decoder.check_threshold_crossings()
    
    def forward(self, x):

        latent = self.encode(x)

        with torch.no_grad():
            binary_latent = (latent >= 0.5).float()
            # binary_latent = (latent >= 0).float()

        loss, trigger = self.decoder(latent + binary_latent - latent.detach(), x)
        return binary_latent, loss, trigger

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