import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hidden_state_dataset import *
import wandb
import os
import time
import math
from SAEs.binary_SAE_fixed import BinarySAEFixed
import numpy as np

class TrainerFixed():
    def __init__(self, config, no_log=False, proj_name=None):
        self.config = config
        if not no_log:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("GPU not available, using CPU")
        else:
            self.device = "cpu"

        # Create model with consistent gamma parameter
        self.model = BinarySAEFixed(
            self.config["input_dim"], 
            self.config["hidden_dim"], 
            self.config["n_bits"],
            self.config["gamma"]
        ).to(self.device)

        self.epoch = 0
        self.chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]

        self.model_path = f"SAEs/b_sae_fixed_{self.config['hidden_dim']}.pth"

        # Initialize W&B
        self.no_log = no_log
        if not self.no_log:
            wandb.init(project=proj_name, config=config)
            wandb.watch(self.model, log="all", log_freq=256)

    def one_epoch(self, dataset, wandb, no_log=False):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        
        # Add learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(dataset) // self.config["batch_size"], 
            eta_min=self.config["lr"] * 0.1
        )

        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        total_loss = 0
        total_activated = 0
        num_batches = 0

        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Check if batch contains NaN before forward pass
            if torch.isnan(batch).any():
                print(f"Batch {batch_idx} contains NaN values before forward pass!")
                continue

            # Forward pass
            latent, recon_loss = self.model(batch)

            # Calculate sparsity (activated neurons)
            activated_neurons = latent.sum(dim=-1).mean()

            # Add L1 sparsity loss if needed
            sparsity_loss = self.config["sparsity_lambda"] * activated_neurons
            
            # Total loss
            loss = recon_loss + sparsity_loss
            
            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()

            # Accumulate statistics
            total_loss += loss.item()
            total_activated += activated_neurons.item()
            num_batches += 1

            if not no_log and batch_idx % 100 == 0:
                # Log metrics
                wandb.log({
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "sparsity_loss": sparsity_loss.item(),
                    "activated_neurons": activated_neurons.item(),
                    "learning_rate": scheduler.get_last_lr()[0],
                    "batch": batch_idx
                })
                
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, Recon={recon_loss.item():.4f}, Activated={activated_neurons.item():.1f}")

        avg_loss = total_loss / num_batches
        avg_activated = total_activated / num_batches
        print(f"Epoch average: Loss={avg_loss:.4f}, Activated neurons={avg_activated:.1f}")

        return self.model
    
    def train(self):
        total_start = time.perf_counter()

        for epoch, f in enumerate(self.chunk_files):
            print(f"\nEpoch {epoch}: Training on {f}")
            
            dataset = HiddenStatesTorchDatasetInBinary(
                os.path.join("dataset/", f), 
                self.config["gamma"], 
                self.config["n_bits"]
            )
            
            self.one_epoch(dataset, wandb, no_log=self.no_log)
            
            # Save checkpoint every few epochs
            if epoch % 5 == 0:
                checkpoint_path = f"{self.model_path}.epoch{epoch}"
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved to {checkpoint_path}")

        if not self.no_log:
            wandb.finish()

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Training completed. Model saved to {self.model_path}")
        total_time = time.perf_counter() - total_start
        print(f"Total training time: {total_time:.2f} seconds")


# Configuration
config = {
    "input_dim": 512,
    "n_bits": 8,
    "hidden_dim": 2048 * 8,  # 16384
    "gamma": 2,
    "epochs": 1,
    "lr": 3e-4,  # Slightly lower learning rate for stability
    "sparsity_lambda": 1e-5,  # Adjusted sparsity penalty
    "batch_size": 512
}

# Train the model
no_log = False
trainer = TrainerFixed(config, no_log, "binary_sae_fixed")
trainer.train() 