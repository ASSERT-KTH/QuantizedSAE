import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from hidden_state_dataset import *
import wandb
import os
import time
import math
from ternary_SAE import *
from binary_latent_SAE import *
from binary_SAE import *

class Trainer():

    def __init__(self, config, sae_type, rigL=False, no_log=False, proj_name=None):
        self.config = config
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        else:
            self.device = torch.device("cpu")
            print("GPU not available, using CPU")
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # self.device = "cpu"

        self.sae_type = sae_type

        if sae_type == "t_sae":
            self.model = TernarySparseAutoencoder(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "bl_sae":
            self.model = BinaryLatentSAE(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "b_sae":
            self.model = BinarySAE(self.config["input_dim"], self.config["hidden_dim"], self.config["n_bits"]).to(self.device)

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
            wandb.watch(self.model, log="all", log_freq=10)

    def one_epoch(self, dataset, wandb, dead_neuron_threshold=0.2, no_log=False, rigL=False, f_decay=None):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])

        dataloader = DataLoader(dataset, batch_size=self.config["batch_size"], shuffle=True, num_workers=4)

        latent, recon, carry = None, None, None

        for batch in dataloader:
            batch = batch.to(self.device)

            if self.sae_type == 'b_sae':
                latent, recon, carry = self.model(batch)

                scale_factor = torch.pow(2, torch.arange(self.config["n_bits"])).to(self.device)
                scale_factor = scale_factor / scale_factor.sum().float()

                batch = batch.view(self.config["batch_size"], self.config["input_dim"], self.config["n_bits"])
                recon = recon.view(self.config["batch_size"], self.config["input_dim"], self.config["n_bits"])
                carry = carry.view(self.config["batch_size"], self.config["input_dim"], self.config["n_bits"])

                recon_loss = torch.mean((((batch - recon) * scale_factor) ** 2).sum(dim=-1))
                # Mean square error is too small
                # recon_loss = (((batch - recon) * scale_factor) ** 2).sum()
                carry *= scale_factor
                sparsity_loss = torch.mean(latent.sum(dim=-1))

            else:
                recon_loss = F.mse_loss(recon, batch)
                latent, recon = self.model(batch)
                sparsity_loss = torch.mean(torch.abs(latent).sum(dim=-1))


            # if epoch < 2:
            #     loss = recon_loss
            # else:
            #     pass
            loss = recon_loss + self.config["sparsity_lambda"] * sparsity_loss

            if self.sae_type == "b_sae":
                carry_loss = torch.mean(carry)
                # carry_loss = self.config["carry_lambda"] * torch.mean(carry ** 2)
                # loss +=  carry_loss

            optimizer.zero_grad()
            loss.backward()

            if rigL == True:
                self.model.decoder.mask_grad()

            optimizer.step()

            # For binary latent:
            # inactivated_neurons = (latent < dead_neuron_threshold).sum(dim=1).float().mean().item()
            inactivated_neurons = torch.mean(latent.sum(dim=-1))

            # For ReLU:
            # dead_neurons = (h == 0).sum(dim=1).float().mean().item()

            if not no_log:
                # Log metrics
                if self.sae_type == "b_sae":
                    wandb.log({
                        "loss": loss,
                        "recon_loss": recon_loss.item(),
                        "sparsity_loss": sparsity_loss.item(),
                        "carry_loss": carry_loss.item(),
                        "inactivated_neurons": inactivated_neurons
                        # "dead_neurons": dead_neurons
                    })
                else:
                    wandb.log({
                        "loss": loss,
                        "recon_loss": recon_loss.item(),
                        "sparsity_loss": sparsity_loss.item(),
                        "inactivated_neurons": inactivated_neurons
                        # "dead_neurons": dead_neurons
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
    "hidden_dim": 512,
    "gamma": 4,
    "n_bits": 4,
    "epochs": 1,
    "lr": 1e-3,
    "sparsity_lambda": 1e-5,
    "carry_lambda": 1e-6,
    "batch_size": 512
}

no_log = False
trainer = Trainer(config, "b_sae", False, no_log, "binary_sae_training_no_carry_loss")

trainer.train()
