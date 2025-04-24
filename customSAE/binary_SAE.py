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

        # nn.init.kaiming_normal_(self.weight)
        nn.init.normal_(self.weight, mean=0.5, std=1.0)
        # Clamp outliers to [0, 1]
        self.weight.data.clamp_(0, 1)

    def forward(self, x):
        # Binary:
        with torch.no_grad():
            self.weight.data.clamp_(0, 1)
            hard_weights = self.weight + ((self.weight >= self.threshold).float() - self.weight).detach() # Binary SAE
            x = x.unsqueeze(-1)
        
        filtered_features = (x * hard_weights.unsqueeze(0))
        print(filtered_features)

        features_by_neurons = torch.split(filtered_features, self.n_bits, dim=-1)
        print(features_by_neurons)

        features_by_neurons_stack = torch.stack(features_by_neurons, dim=-3)
        features_by_neurons_stack_reshape = features_by_neurons_stack.reshape(-1, features_by_neurons_stack.shape[-2], features_by_neurons_stack.shape[-1])

        sum, carry = self.csa(features_by_neurons_stack_reshape)

        sum = sum.reshape(-1, self.n_bits*self.out_features)

        return sum, carry

class binary_SAE(SparseAutoencoder):

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

        return self.decode(latent + (binary_latent - latent).detach())

# bd = binary_decoder(3, 2, 2)
# 
# testing_case = torch.tensor([[1, 0, 1], [0, 1, 0]])
# # testing_case = torch.tensor([[1, 0, 1]])
# print(bd(testing_case))

# bin_sae = binary_SAE(2, 4, 2)
# 
# testing_case = torch.tensor([[1., 0., 1., 0.], [0., 1., 1., 1.]])
# print(bin_sae(testing_case))