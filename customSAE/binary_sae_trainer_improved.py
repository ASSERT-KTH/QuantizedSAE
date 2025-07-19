import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
from SAEs.binary_SAE import BinarySAE
import wandb
import os
import time

# Custom initialization for binary SAE
def init_binary_sae(model):
    """Initialize binary SAE to prevent activation collapse."""
    # Encoder initialization
    with torch.no_grad():
        # Use larger initialization for encoder weights
        if hasattr(model.encoder[0], 'weight_g'):
            # Weight normalized layer
            nn.init.xavier_uniform_(model.encoder[0].weight_v)
            nn.init.constant_(model.encoder[0].weight_g, 1.0)
        else:
            # Regular layer
            nn.init.xavier_uniform_(model.encoder[0].weight)
        
        # Initialize bias to encourage ~10% activation initially
        if hasattr(model.encoder[0], 'bias'):
            nn.init.constant_(model.encoder[0].bias, -1.5)  # sigmoid(-1.5) ≈ 0.18
    
    return model

class ImprovedBinaryTrainer:
    def __init__(self, config, no_log=False, proj_name=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create and initialize model
        self.model = BinarySAE(
            config["input_dim"], 
            config["hidden_dim"], 
            config["n_bits"]
        ).to(self.device)
        
        # Apply custom initialization
        self.model = init_binary_sae(self.model)
        
        self.chunk_files = [f for f in os.listdir("dataset/") 
                           if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
        
        self.no_log = no_log
        if not self.no_log:
            wandb.init(project=proj_name, config=config)
    
    def train_epoch(self, dataset):
        # Use lower learning rate and gradient clipping
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config["lr"])
        
        # Warmup scheduler to prevent early gradient explosion
        warmup_steps = 100
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda step: min(1.0, step / warmup_steps)
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.config["batch_size"], 
            shuffle=True, 
            num_workers=4
        )
        
        for batch_idx, batch in enumerate(dataloader):
            batch = batch.to(self.device)
            
            # Forward pass
            latent, recon_loss = self.model(batch)
            
            # Monitor activation statistics
            with torch.no_grad():
                activation_rate = latent.mean().item()
                activated_neurons = latent.sum(dim=-1).mean().item()
                
                # Warning if activation collapse
                if activation_rate < 0.05 and batch_idx > warmup_steps:
                    print(f"⚠️  Warning: Low activation rate: {activation_rate:.4f}")
            
            # Add sparsity loss with schedule
            sparsity_target = self.config["sparsity_target"]  # e.g., 0.05 for 5% activation
            sparsity_loss = self.config["sparsity_lambda"] * (latent.mean() - sparsity_target).abs()
            
            # Total loss
            loss = recon_loss + sparsity_loss
            
            # Backward pass with gradient clipping
            optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients to prevent explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Check gradient norms before step
            with torch.no_grad():
                total_norm = 0
                for p in self.model.parameters():
                    if p.grad is not None:
                        total_norm += p.grad.norm().item() ** 2
                total_norm = total_norm ** 0.5
                
                if total_norm > 100:
                    print(f"⚠️  Large gradient norm: {total_norm:.2f}")
            
            optimizer.step()
            scheduler.step()
            
            # Logging
            if not self.no_log and batch_idx % 100 == 0:
                wandb.log({
                    "loss": loss.item(),
                    "recon_loss": recon_loss.item(),
                    "sparsity_loss": sparsity_loss.item(),
                    "activation_rate": activation_rate,
                    "activated_neurons": activated_neurons,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "gradient_norm": total_norm
                })
                
                print(f"Batch {batch_idx}: Loss={loss.item():.4f}, "
                      f"Act rate={activation_rate:.4f}, "
                      f"Activated={activated_neurons:.1f}")
    
    def train(self):
        for epoch, f in enumerate(self.chunk_files):
            print(f"\nEpoch {epoch}: Training on {f}")
            
            dataset = HiddenStatesTorchDatasetInBinary(
                os.path.join("dataset/", f),
                self.config["gamma"],
                self.config["n_bits"]
            )
            
            self.train_epoch(dataset)
            
            # Save checkpoint
            if epoch % 5 == 0:
                torch.save(
                    self.model.state_dict(), 
                    f"SAEs/b_sae_improved_{self.config['hidden_dim']}_epoch{epoch}.pth"
                )
        
        if not self.no_log:
            wandb.finish()

# Configuration with improvements
config = {
    "input_dim": 512,
    "n_bits": 8,
    "hidden_dim": 2048 * 8,
    "gamma": 2,
    "lr": 1e-4,  # Lower learning rate
    "sparsity_lambda": 1e-4,
    "sparsity_target": 0.05,  # Target 5% activation
    "batch_size": 256  # Smaller batch size for stability
}

if __name__ == "__main__":
    trainer = ImprovedBinaryTrainer(config, no_log=False, proj_name="binary_sae_improved")
    trainer.train() 