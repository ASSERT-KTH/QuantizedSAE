import torch
import torch.nn as nn
import torch.nn.functional as F
from baseSAE.SAE import SparseAutoencoder

class BinaryLatentSAE(SparseAutoencoder):

    def __init__(self, input_dim, hidden_dim):

        super().__init__(input_dim, hidden_dim)

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

# torch.manual_seed(42)
# 
# bl_sae = BinaryLatentSAE(2, 2)
# 
# testing_cases = torch.tensor([[2, 2], [3, 3]]).float()
# 
# binary_latent, recon = bl_sae(testing_cases)
# loss = F.mse_loss(recon, testing_cases)
# loss.backward()
# 
# for name, param in bl_sae.named_parameters():
#     if param.requires_grad and param.grad is not None:
#         print(f"grad_norms/{name}: {param.grad.norm()}")
#         print(f"grad_values/{name}: {param.grad}")