import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ..data.dataset import HiddenStatesTorchDataset
import wandb  # Will be imported conditionally if needed
import os
import time
import math
from ..sae.ternary import TernarySparseAutoencoder
from ..sae.binary_latent import BinaryLatentSAE
from ..sae.binary import BinarySAE
from ..sae.quantized_matryoshka import QuantizedMatryoshkaSAE
from ..sae.residual_quantized import ResidualQuantizedSAE
from ..sae.baseline import BaselineSparseAutoencoder
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
        elif sae_type == "q_sae":
            self.model = QuantizedMatryoshkaSAE(self.config["input_dim"], self.config["hidden_dim"], self.config["top_k"], self.config["gamma"], self.config["n_bits"]).to(self.device)
        elif sae_type == "rq_sae":
            self.model = ResidualQuantizedSAE(self.config["input_dim"], self.config["hidden_dim"], self.config["top_k"], self.config["gamma"], self.config["n_bits"]).to(self.device)
        elif sae_type == "baseline_sae":
            self.model = BaselineSparseAutoencoder(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)

        self.epoch = 0
        self.chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]

        # rigL settings:
        self.rigL = rigL
        self.connection_fraction_to_update = 0.3
        self.f_decay = None

        self.model_path = "SAEs/" + sae_type + "_" + str(self.config["hidden_dim"]) + ("_rigL" if rigL else "") + (str(self.config["n_bits"]) + "_bits" if sae_type == "b_sae" or sae_type == "q_sae" or sae_type == "rq_sae" else "") + ".pth"

        # Initialize W&B
        self.no_log = no_log
        if not self.no_log:
            wandb.init(project=proj_name, config=config)
            wandb.watch(self.model, log="all", log_freq=256)

    def one_epoch(self, dataset, dead_neuron_threshold=0.2, no_log=False, rigL=False, f_decay=None):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        # optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config["lr"], weight_decay=1e-4)
        # optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.config["lr"], alpha=0.9)

        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=0)

        latent, recon, carry = None, None, None
        batch_idx = 0

        latents = []
        for batch in dataloader:
            batch_idx += 1
            batch = batch.to(self.device)
            
            # Check if batch contains NaN before forward pass
            if torch.isnan(batch).any():
                print(f"Batch {batch_idx} contains NaN values before forward pass!")
                continue

            if self.sae_type == "q_sae":
                latent_group, recon_groups = self.model(batch)  # list of length n_bits, each [B, input_dim]

                optimizer.zero_grad(set_to_none=True)

                recon_losses = []
                prev = torch.zeros_like(batch)
                for i, recon in enumerate(recon_groups):
                    # Residual loss:
                    # inc = recon - prev
                    # target_residual = (batch - prev).detach()
                    # losses.append(F.mse_loss(inc, target_residual))
                    # prev = recon

                    # Total loss:
                    # losses.append(F.mse_loss(recon, batch) * 2 ** (i + 2) / self.config["gamma"])
                    recon_losses.append(0.5*F.mse_loss(recon, batch))

                sparsity_loss = sum(latent_group) * self.config["sparsity_lambda"]
                recon_loss = sum(recon_losses)
                loss_total = recon_loss + sparsity_loss
                loss_total.backward()

                self.model.decoder.apply_secant_grad()
                optimizer.step()
            
            elif self.sae_type == "rq_sae":
                latent_group, recon_group = self.model(batch)
                residual = batch
                optimizer.zero_grad(set_to_none=True)
                recon_losses = []

                sparsity_loss = 0

                for i, recon in enumerate(recon_group):
                    recon_losses.append(0.5*F.mse_loss(recon, residual))
                    # residual = (residual - recon).detach()
                    residual = (residual - recon).detach() * 2
                    # sparsity_loss += latent_group[i] * self.config["sparsity_lambda"] * 2
                    if i == 0:
                        sparsity_loss += latent_group[i] * self.config["sparsity_lambda"]
                    elif i == 1:
                        sparsity_loss += latent_group[i] * self.config["sparsity_lambda"] * 2.5 # Previous: 4, 6
                    elif i == 2:
                        sparsity_loss += latent_group[i] * self.config["sparsity_lambda"] * 4 # Previous: 8, 16, 32, 80
                    elif i == 3:
                        sparsity_loss += latent_group[i] * self.config["sparsity_lambda"] * 8 # Previous: 16, 32, 64, 256
                recon_loss = sum(recon_losses)
                # sparsity_loss = sum(latent_group) * self.config["sparsity_lambda"]
                loss_total = recon_loss + sparsity_loss

                loss_total.backward()

                self.model.apply_secant_grad()
                optimizer.step()

            elif self.sae_type == 'b_sae':

                latent, reconstruction, polarize_loss = self.model(batch)

                active_per_sample = latent.sum(dim=1)
                inactive_per_sample = self.config["hidden_dim"] - active_per_sample

                recon_loss = 0.5 * F.mse_loss(reconstruction, batch)
                loss = recon_loss + self.config["polarize_lambda"] * polarize_loss
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            elif self.sae_type == "t_sae":
                latent, recon = self.model(batch)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                self.model.decoder.mask_grad()
                optimizer.step()
                self.model.decoder.update_mask(self.f_decay, 0.7)
            
            elif self.sae_type == "baseline_sae":

                latent, recon = self.model(batch)
                loss = F.mse_loss(recon, batch)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.model.normalize_decoder_weights()

            if not no_log and batch_idx % 100 == 0:
                # Log metrics
                if self.sae_type == "b_sae":
                    wandb.log({
                        "loss": loss.item(),
                        "recon_loss": recon_loss.item(),
                        "polarize_loss": polarize_loss.item(),
                        "activated_neurons": torch.mean(latent.sum(dim=-1)).item(),
                        "mag_MSB": self.model.decoder.weight[:, self.config["n_bits"]-1::self.config["n_bits"]].abs().mean().item(),
                        "mag_LSB": self.model.decoder.weight[:, 0::self.config["n_bits"]].abs().mean().item(),
                    })
                elif self.sae_type == "q_sae":
                    log_dict = {f"recon_loss_group_{i}": recon_losses[i].item() for i in range(self.config["n_bits"])}
                    log_dict["recon_loss_total"] = recon_loss.item()
                    log_dict.update({f"L0 of latent_group_{i}": latent_group[i].item() for i in range(self.config["n_bits"])})
                    log_dict.update({"sparsity loss": sparsity_loss.item()})
                    wandb.log(log_dict)
                elif self.sae_type == "rq_sae":
                    log_dict = {f"recon_loss_group_{i}": recon_losses[i].item() / 4**i for i in range(self.config["n_bits"])}
                    log_dict.update({f"L0 of latent_group_{i}": latent_group[i].item() for i in range(self.config["n_bits"])})
                    log_dict.update({"sparsity loss": sparsity_loss.item()})
                    wandb.log(log_dict)
                elif self.sae_type == "t_sae":
                    wandb.log({
                        "loss": loss.item()
                    })
                elif self.sae_type == "baseline_sae":
                    wandb.log({
                        "loss": loss.item()
                    })
                else:
                    wandb.log({
                        "loss": loss.item(),
                        # "recon_loss": recon_loss.item(),
                        "sparsity_loss": sparsity_loss.item()
                        # "inactivated_neurons": inactivated_neurons.item() if isinstance(inactivated_neurons, torch.Tensor) else inactivated_neurons
                    })

            if no_log and batch_idx % 100 == 0:
                # Log metrics
                if self.sae_type == "b_sae":
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f}, recon_loss={recon_loss.item():.4f}, polarize_loss={polarize_loss.item():.4f}, activated_neurons={torch.mean(latent.sum(dim=-1)).item():.4f}, mag_MSB={self.model.decoder.weight[:, self.config['n_bits']-1::self.config['n_bits']].abs().mean().item():.4f}, mag_LSB={self.model.decoder.weight[:, 0::self.config['n_bits']].abs().mean().item():.4f}")
                elif self.sae_type == "q_sae":
                    recon_losses_str = ", ".join([f"recon_loss_group_{i}={recon_losses[i].item():.4f}" for i in range(self.config['n_bits'])])
                    l0_str = ", ".join([f"L0_of_latent_group_{i}={latent_group[i].item():.4f}" for i in range(self.config['n_bits'])])
                    print(f"Batch {batch_idx}: {recon_losses_str}, recon_loss_total={loss_total.item():.4f}, {l0_str}")
                elif self.sae_type == "rq_sae":
                    recon_losses_str = ", ".join([f"recon_loss_group_{i}={recon_losses[i].item()/4**i:.4f}" for i in range(self.config['n_bits'])])
                    l0_str = ", ".join([f"L0_of_latent_group_{i}={latent_group[i].item():.4f}" for i in range(self.config['n_bits'])])
                    print(f"Batch {batch_idx}: {recon_losses_str}, recon_loss_total={loss_total.item():.4f}, {l0_str}")
                elif self.sae_type == "t_sae":
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f}")
                elif self.sae_type == "baseline_sae":
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f}")
                else:
                    print(f"Batch {batch_idx}: Loss={loss.item():.4f}, sparsity_loss={sparsity_loss.item():.4f}")

        return self.model
    
    def train(self):

        total_start = time.perf_counter()

        for epoch, f in enumerate(self.chunk_files):
            if epoch > 100:
                break
            print(f"Training on {f}:")
            # if self.sae_type == "b_sae":
            #     dataset = HiddenStatesTorchDatasetInBinary(os.path.join("dataset/", f), self.config["gamma"], self.config["n_bits"])
            # else:
            #     dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
            dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
            # print(f"The dataset size is {dataset.__len__()}.")
            if self.rigL:
                self.f_decay = self.connection_fraction_to_update / 2 * (1 + math.cos(epoch*math.pi/len(self.chunk_files)))
                self.model.decoder.update_mask(self.f_decay, 0.7)
            else:
                f_decay = None
            self.one_epoch(dataset, dead_neuron_threshold=0.2, no_log=self.no_log, rigL=self.rigL, f_decay=self.f_decay)

        if not self.no_log:
            wandb.finish()

        torch.save(self.model.state_dict(), self.model_path)
        print(f"Training completed. Model been saved to {self.model_path}.")
        total_time = time.perf_counter() - total_start
        print(f"Total training time: {total_time:.2f} seconds")

# Configuration
config = {
    "input_dim": 512,
    "n_bits": 4,
    # "hidden_dim": 2048,
    "hidden_dim": 2048 * 16,
    "gamma": 1.5,
    "epochs": 1,
    "lr": 1e-4,
    "top_k": 32,
    "sparsity_lambda": 1.5*1e-3,
    "polarize_lambda": 1e-2,
    "batch_size": 1024*8
}

# no_log = True
no_log = False
# trainer = Trainer(config, "b_sae", False, no_log, "binary_sae_training_no_carry_loss")
# trainer = Trainer(config, "q_sae", False, no_log, "quantized_matryoshka_sae_training")
trainer = Trainer(config, "baseline_sae", False, no_log, "baseline_sae_training")
# trainer = Trainer(config, "t_sae", True, no_log, "ternary_sae_training")
# trainer = Trainer(config, "baseline_sae", False, no_log, "baseline_sae_training")

trainer.train()
