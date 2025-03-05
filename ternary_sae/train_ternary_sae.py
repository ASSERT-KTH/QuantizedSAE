import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import wandb
import os
import time
from ternary_sae import *

def train(model, dataset, config, wandb, epoch, no_log=False):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
    for batch in dataloader:
        batch = batch.to(device)
        
        h, recon = model(batch)
        
        recon_loss = F.mse_loss(recon, batch)
        sparsity_loss = torch.mean(torch.abs(h))
        if epoch < 2:
            loss = recon_loss
        else:
            loss = recon_loss + config["sparsity_lambda"] * sparsity_loss
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dead_neurons = (h == 0).sum(dim=1).float().mean().item()

        if not no_log:
            # Log metrics
            wandb.log({
                "loss": loss,
                "recon_loss": recon_loss.item(),
                "sparsity_loss": sparsity_loss.item(),
                "dead_neurons": dead_neurons
            })
        
    return model

# Configuration
config = {
    "input_dim": 512,
    "hidden_dim": 4096,
    "epochs": 1,
    "lr": 1e-3,
    "sparsity_lambda": 1e-4,
    "batch_size": 8192
}

hidden_dim = config["hidden_dim"]

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

model = TernarySparseAutoencoder(config["input_dim"], config["hidden_dim"]).to(device)

model_path = f"SAEs/t_sae_hidden_{hidden_dim}.pth"
if os.path.exists(model_path):
    print(f"{model_path} exists.")
    model.load_state_dict(torch.load(model_path)).to(device)

chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]

no_log = False

# Initialize W&B
if not no_log:
    wandb.init(project="ternary_sae", config=config)
    wandb.watch(model, log="all", log_freq=1000)

# Start training
total_start = time.perf_counter()
for epoch, f in enumerate(chunk_files):
    print(f"Training on {f}:")
    dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
    train(model, dataset, config, wandb, epoch, no_log)

if not no_log:
    wandb.finish()
torch.save(model.state_dict(), model_path)
print(f"Training completed. Model been saved to {model_path}.")
total_time = time.perf_counter() - total_start
print(f"Total training time: {total_time:.2f} seconds")