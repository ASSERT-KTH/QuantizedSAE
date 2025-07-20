import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hidden_state_dataset import *
import wandb
import os
import time
import math
from SAEs.ternary_SAE import *
from SAEs.binary_latent_SAE import *
from SAEs.binary_SAE import *
import numpy as np

class Trainer():

    def __init__(self, config, sae_type, rigL=False, no_log=False, proj_name=None):
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

        self.sae_type = sae_type

        if sae_type == "t_sae":
            self.model = TernarySparseAutoencoder(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "bl_sae":
            self.model = BinaryLatentSAE(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "b_sae":
            self.model = BinarySAE(self.config["input_dim"], self.config["hidden_dim"], self.config["n_bits"]).to(self.device)
            self.scale_factor = torch.pow(2, torch.arange(self.config["n_bits"])).to(self.device)
            self.scale_factor = self.scale_factor / self.scale_factor.sum().float()
            # self.model = torch.compile(self.model, mode="reduce-overhead")

        self.epoch = 0
        self.chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]

        # rigL settings:
        self.rigL = rigL
        self.connection_fraction_to_update = 0.3

        self.model_path = "SAEs/" + sae_type + "_" + str(self.config["hidden_dim"]) + "rigL" if rigL else "" + ".pth"

        # Initialize W&B
        self.no_log = no_log
        if not self.no_log:
            wandb.init(project=proj_name, config=config)
            wandb.watch(self.model, log="all", log_freq=256)

    def one_epoch(self, dataset, wandb, dead_neuron_threshold=0.2, no_log=False, rigL=False, f_decay=None):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        
        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=len(dataset) // self.config["batch_size"], 
            eta_min=self.config["lr"] * 0.1
        )

        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config["lr"], alpha=0.9)

        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        latent, recon, carry = None, None, None
        batch_idx = 0
        
        # Track losses for adaptive weighting
        recon_losses = []
        polarize_losses = []

        latents = []
        for batch in dataloader:
            batch_idx += 1
            batch = batch.to(self.device)
            
            # Check if batch contains NaN before forward pass
            if torch.isnan(batch).any():
                print(f"Batch {batch_idx} contains NaN values before forward pass!")
                continue

            if self.sae_type == 'b_sae':

                # self.model.store_decoder_pre_update_state()
                latent, recon_loss, polarize_loss = self.model(batch)

                active_per_sample = latent.sum(dim=1)
                inactive_per_sample = self.config["hidden_dim"] - active_per_sample

                # sparsity_loss = self.config["sparsity_lambda"] * torch.mean(active_per_sample)
                sparsity_loss = torch.tensor(0.)
                
                # Adaptive loss weighting
                if len(recon_losses) > 10:
                    # Calculate running average of losses
                    avg_recon = sum(recon_losses[-10:]) / 10
                    avg_polarize = sum(polarize_losses[-10:]) / 10
                    
                    # Adjust polarize weight based on relative magnitudes
                    polarize_weight = min(0.1, avg_recon / (avg_polarize + 1e-8) * 0.01)
                else:
                    polarize_weight = 0.01  # Start with small weight
                
                recon_losses.append(recon_loss.item())
                polarize_losses.append(polarize_loss.item())

                # loss = recon_loss + sparsity_loss
                loss = recon_loss + polarize_weight * polarize_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Check for gradient issues
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                optimizer.step()
                scheduler.step()

                # self.model.check_decoder_threshold_crossings()

                # For binary latent:
                # inactivated_neurons = (latent < dead_neuron_threshold).sum(dim=1).float().mean().item()
                inactivated_neurons = self.config["hidden_dim"] - torch.mean(latent.sum(dim=-1))

                # For ReLU:
                # dead_neurons = (h == 0).sum(dim=1).float().mean().item()
                print(f"Batch {batch_idx}: recon_loss={recon_loss:.4f}, polarize_loss={polarize_loss:.4f}, grad_norm={total_norm:.4f}")
                print(f"Active neurons: {torch.mean(latent.sum(dim=-1)):.1f}")

            if not no_log and batch_idx % 100 == 0:
                # Log metrics
                if self.sae_type == "b_sae":
                    wandb.log({
                        "loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "polarize_loss": polarize_loss.item(),
                        "polarize_weight": polarize_weight,
                        "sparsity_loss": sparsity_loss.item(),
                        "activated_neurons": torch.mean(latent.sum(dim=-1)).item(),
                        "mag_MSB": self.model.decoder.weight[:, self.config["n_bits"]-1::self.config["n_bits"]].abs().mean().item(),
                        "mag_LSB": self.model.decoder.weight[:, 0::self.config["n_bits"]].abs().mean().item(),
                        "gradient_norm": total_norm,
                        "learning_rate": scheduler.get_last_lr()[0],
                        "temperature": self.model.temperature.item() if hasattr(self.model, 'temperature') else 1.0,
                        # "inactive_mean" : inactive_per_sample.float().mean(),
                        # "inactive_std"  : inactive_per_sample.float().std(),  # <= new!
                    })
                else:
                    wandb.log({
                        "loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "sparsity_loss": sparsity_loss.item(),
                        "inactivated_neurons": inactivated_neurons.item() if isinstance(inactivated_neurons, torch.Tensor) else inactivated_neurons
                    })

        return self.model
    
    def train(self):

        total_start = time.perf_counter()

        for epoch, f in enumerate(self.chunk_files):
            print(f"Training on {f}:")
            if self.sae_type == "b_sae":
                dataset = HiddenStatesTorchDatasetInBinary(os.path.join("dataset/", f), self.config["gamma"], self.config["n_bits"])
            else:
                dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
            # print(f"The dataset size is {dataset.__len__()}.")
            if self.rigL:
                f_decay = self.connection_fraction_to_update / 2 * (1 + math.cos(epoch*math.pi/len(self.chunk_files)))
                self.model.decoder.update_mask(f_decay, 0.7)
            else:
                f_decay = None
            self.one_epoch(dataset, wandb, dead_neuron_threshold=0.2, no_log=self.no_log, rigL=self.rigL, f_decay=f_decay)

        if not self.no_log:
            wandb.finish()

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Training completed. Model been saved to {self.model_path}.")
        total_time = time.perf_counter() - total_start
        print(f"Total training time: {total_time:.2f} seconds")

# Configuration
config = {
    "input_dim": 512,
    "n_bits": 8,
    # "hidden_dim": 2048,
    "hidden_dim": 2048 * 8,
    "gamma": 2,
    "epochs": 1,
    "lr": 5e-4,  # Reduced learning rate for stability
    "sparsity_lambda": 1e-6,
    "carry_lambda": 1e-6,
    "batch_size": 512  # Smaller batch size for better gradient estimates
}

# no_log = True
no_log = False
trainer = Trainer(config, "b_sae", False, no_log, "binary_sae_training_no_carry_loss")

trainer.train()
