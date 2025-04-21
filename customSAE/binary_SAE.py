import torch
import torch.nn as nn
import torch.nn.functional as F
from baseSAE.SAE import SparseAutoencoder
from nnba.adder import *

class binary_decoder(nn.Module):
    def __init__(self, in_features, out_features, n_bits=8):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features*n_bits))
        self.threshold = 0.5 # For binary SAE

        # nn.init.kaiming_normal_(self.weight)
        nn.init.normal_(self.weight, mean=0.5, std=1.0)
        # Clamp outliers to [0, 1]
        self.weight.data.clamp_(0, 1)

    def forward(self, x):
        # Binary:
        with torch.no_grad():
            self.weight = torch.clamp(self.weight, 0, 1)
            hard_weights = self.weight + ((self.weight >= self.threshold).float() - self.weight).detach() # Binary SAE
            x = x.unsqueeze(-1)
        
        filtered_features = x * hard_weights.unsqueeze(0)

        pass

class binary_SAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim, n_bits=8):

        super().__init__(input_dim, hidden_dim)

        self.n_bits = n_bits

        self.encoder = nn.Sequential(
            nn.Linear(input_dim*self.n_bits, hidden_dim),
            nn.Sigmoid()
        )

        self.decoder = binary_decoder(hidden_dim, input_dim, n_bits=self.n_bits)