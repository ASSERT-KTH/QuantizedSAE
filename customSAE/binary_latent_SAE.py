import torch
import torch.nn as nn
import torch.nn.functional as F
from ..baseSAE import SAE

def BinaryLatentSAE(SAE):

    def __init__(self, input_dim, hidden_dim):

        super.__init__(input_dim, hidden_dim)

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        latent = self.encode(x)
        with torch.no_grad():
            binary_latent = (latent >= 0.5).float()

        reconstruction = self.decode(latent + (binary_latent - latent).detach())
        return binary_latent, reconstruction