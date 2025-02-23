import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import time

class STEWeights(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_normal_(self.weight)
        
    def forward(self, x):
        hard_weights = torch.clamp(self.weight, -1, 1)  # Ternary SAE
        # hard_weights = torch.sign(self.weight)  # Binary
        return F.linear(x, hard_weights + (self.weight - self.weight.detach()))

class TernarySparseAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = STEWeights(hidden_dim, input_dim)
        
    def forward(self, x):
        h = self.encoder(x)
        return h, self.decoder(h)

class HiddenStatesTorchDataset(Dataset):
    def __init__(self, file_path, transform=None):
        """
        file_paths: .pt file path. The file is expected to store a tensor 
                    with shape of (num_contexts, tokens_per_context, feature_dim) where
                    feature_dim is expected to be 512.
        transform: Optional transformation to apply to each sample.
        """
        self.data = torch.load(file_path, map_location='cpu')
        self.transform = transform
        num_contexts, tokens_per_context, feature_dim = self.data.shape
        
        self.cum_sizes = num_contexts * tokens_per_context
        self.files_info = (file_path, num_contexts, tokens_per_context, feature_dim)

    def __len__(self):
        return self.cum_sizes

    def __getitem__(self, idx):
        # Map the local index to (context index, token index)
        context_idx = idx // self.files_info[2]
        token_idx = idx % self.files_info[2] 
        sample = self.data[context_idx, token_idx, :]  # Each sample is a 512-d tensor
        
        sample = sample.float()
        return sample

def train(dataset, config):
    # Initialize W&B
    wandb.init(project="ternary_sae", config=config)
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cpu"
    model = TernarySparseAutoencoder(config["input_dim"], config["hidden_dim"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    
    wandb.watch(model, log="all", log_freq=100)

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
    for batch in dataloader:
        batch = batch.to(device)
        
        h, recon = model(batch)
        
        recon_loss = F.mse_loss(recon, batch)
        sparsity_loss = torch.mean(torch.abs(h))
        loss = recon_loss + config["sparsity_lambda"] * sparsity_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Log metrics
        wandb.log({
            "loss": loss/config["batch_size"],
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item()
        })
        
    wandb.finish()
    return model

# Configuration
config = {
    "input_dim": 512,
    "hidden_dim": 512,
    "epochs": 1,
    "lr": 1e-3,
    "sparsity_lambda": 1e-4,
    "batch_size": 128
}

# Start training
total_start = time.perf_counter()

dataset = HiddenStatesTorchDataset("dataset/the_pile_hidden_states_L3_10.pt")
model = train(dataset, config)

total_time = time.perf_counter() - total_start
print(f"Total training time: {total_time:.2f} seconds")
