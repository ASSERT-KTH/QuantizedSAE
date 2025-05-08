import torch
import torch.nn as nn
import torch.nn.functional as F
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class binary_decoder(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_bits = n_bits
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features*n_bits))
        self.threshold = 0.5 # For binary SAE

        self.csa = carry_save_adder(n_bits)

        for param in self.csa.parameters():
            param.requires_grad = False
            # param.requires_grad = True

        # nn.init.kaiming_normal_(self.weight)
        nn.init.normal_(self.weight, mean=0.5, std=1.0)
        # Clamp outliers to [0, 1]
        self.weight.data.clamp_(0, 1)

    def forward(self, x):
        # Binary weights(feature representations):
        # self.weight.data.clamp_(0, 1) # Problematic part
        # clamped_weight = self.weight.clamp(0, 1)
        hard_weights = self.weight + ((self.weight >= self.threshold).float() - self.weight).detach() # Binary SAE
        x = x.unsqueeze(-1)
        
        # Reshape features for batch processing:
        filtered_features = (x * hard_weights.unsqueeze(0))
        # print(filtered_features)

        # Following is the original method, consuming too much memory because of the split and stack operation
        # features_by_neurons = torch.split(filtered_features, self.n_bits, dim=-1)
        # print(features_by_neurons)
        # features_by_neurons_stack = torch.stack(features_by_neurons, dim=-3)

        features_by_neurons = filtered_features.unfold(-1, self.n_bits, self.n_bits).permute(0, 2, 1, 3)
        # Batch: 512, Feature: 512, Neuron: 512, N_bits: 4

        features_by_neurons = features_by_neurons.reshape(-1, features_by_neurons.shape[-2], features_by_neurons.shape[-1])

        sum, carry = self.csa(features_by_neurons)

        sum = sum.reshape(-1, self.n_bits*self.out_features)

        return sum, carry

class BinarySAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, n_bits=8):

        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim*self.n_bits, hidden_dim),
            nn.Sigmoid()
        )

        self.decoder = binary_decoder(hidden_dim, input_dim, n_bits=self.n_bits)
    
    def forward(self, x):

        latent = self.encode(x)

        with torch.no_grad():
            binary_latent = (latent >= 0.5).float()

        recon, carry = self.decode(latent + binary_latent - latent.detach())
        return binary_latent, recon, carry

# bd = binary_decoder(3, 2, 2)
# 
# testing_case = torch.tensor([[1, 0, 1], [0, 1, 0]])
# # testing_case = torch.tensor([[1, 0, 1]])
# print(bd(testing_case))

# torch.manual_seed(42)
#  
# bin_sae = BinarySAE(2, 2, 2)
# 
# testing_case = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 1.]])
# print(bin_sae(testing_case))
# latent, out, carry = bin_sae(testing_case)
# 
# scale_factor = torch.pow(2, torch.arange(2))
# scale_factor = scale_factor / scale_factor.sum().float()
# # scale_factor = scale_factor
# 
# batch = testing_case.view(2, 2, 2)
# recon = out.view(2, 2, 2)
# carry = carry.view(2, 2, 2)
# 
# # recon_loss = torch.mean((((batch - recon) * scale_factor) ** 2).sum(dim=-1))
# recon_loss = (((batch - recon) * scale_factor) ** 2).sum()
# carry *= scale_factor
# sparsity_loss = torch.mean(latent.sum(dim=-1))
# 
# loss = recon_loss + 1e-6 * sparsity_loss
# loss.backward()
# 
# for name, param in bin_sae.named_parameters():
#     if param.requires_grad and param.grad is not None:
#         print(f"grad_norms/{name}: {param.grad.norm()}")
#         print(f"grad_values/{name}: {param.grad}")
# 
# latent = bin_sae.encoder(testing_case)
# latent.sum().backward()
# for name, param in bin_sae.encoder.named_parameters():
#     if param.requires_grad and param.grad is not None:
#         print(f"grad_norms/{name}: {param.grad.norm()}")
#         print(f"grad_values/{name}: {param.grad}")
# 