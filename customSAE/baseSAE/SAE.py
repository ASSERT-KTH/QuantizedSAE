import torch
import torch.nn as nn
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # These should be implemented by subclasses
        self.encoder = None
        self.decoder = None
        
    def encode(self, x):
        if self.encoder is None:
            raise NotImplementedError("Encoder has not been implemented.")
        return self.encoder(x)
        
    def decode(self, h):
        if self.decoder is None:
            raise NotImplementedError("Decoder has not been implemented.")
        return self.decoder(h)
    
    def forward(self, x):
        latent = self.encode(x)
        reconstruction = self.decode(latent)
        return latent, reconstruction
