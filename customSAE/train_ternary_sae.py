import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hidden_state_dataset import HiddenStatesTorchDataset
import wandb
import os
import time
import math
from ternary_SAE import *

def train(model, dataset, config, wandb, epoch, no_log=False, rigL=False, f_decay=None):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4)
    
    for batch in dataloader:
        batch = batch.to(device)
        
        # h, recon = model(batch)
        
        # recon_loss = F.mse_loss(recon, batch)
        recon_loss = F.mse_loss(torch.zeros(batch.shape), batch)
        # sparsity_loss = torch.mean(torch.abs(h))
        if epoch < 2:
            loss = recon_loss
        else:
            pass
           # loss = recon_loss + config["sparsity_lambda"] * sparsity_loss
        
        # optimizer.zero_grad()
        # loss.backward()

        if rigL == True:
            model.decoder.mask_grad()

        # optimizer.step()
        
        # dead_neurons = (h == 0).sum(dim=1).float().mean().item()

        if not no_log:
            # Log metrics
            wandb.log({
                "loss": loss,
                "recon_loss": recon_loss.item(),
                # "sparsity_loss": sparsity_loss.item(),
                # "dead_neurons": dead_neurons
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

model_path = f"SAEs/t_sae_hidden_{hidden_dim}_rigL_by_chunk.pth"
# if os.path.exists(model_path):
#     print(f"{model_path} exists.")
#     model.load_state_dict(torch.load(model_path)).to(device)

chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]

no_log = False

# Initialize W&B
if not no_log:
    wandb.init(project="ternary_sae_rigL_by_chunk", config=config)
    wandb.watch(model, log="all", log_freq=1000)

# Start training
rigL = False
connection_fraction_to_update = 0.3
total_start = time.perf_counter()
if rigL:
    model.decoder.init_mask(0.7)
for epoch, f in enumerate(chunk_files):
    print(f"Training on {f}:")
    dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
    f_decay = connection_fraction_to_update / 2 * (1 + math.cos(epoch*math.pi/len(chunk_files)))
    model.decoder.update_mask(f_decay, 0.7)
    train(model, dataset, config, wandb, epoch, no_log, rigL=rigL, f_decay=f_decay)

if not no_log:
    wandb.finish()

torch.save(model.state_dict(), model_path)
print(f"Training completed. Model been saved to {model_path}.")
total_time = time.perf_counter() - total_start
print(f"Total training time: {total_time:.2f} seconds")