import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from datetime import datetime

from SAEs.binary_SAE import *

# Set up device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cuda"
device = "cpu"
print(f"Using device: {device}")

def generate_binary_latents(batch_size=512):
    return torch.bernoulli(torch.ones(batch_size, 512) * 0.5).to(device)

def train_step(optimizer, latent_z, target_x):
    optimizer.zero_grad()
    
    pred_x, carry = fake_decoder(latent_z)
    
    scale_factor = torch.pow(2, torch.arange(4, device=device))
    scale_factor = scale_factor / scale_factor.sum()
    target_x = target_x.view(512, 512, 4) * scale_factor
    pred_x = pred_x.view(512, 512, 4) * scale_factor
    carry = carry.view(512, 512, 4)
    
    msb_carry = carry[:, :, -1]
    carry_loss = torch.mean(msb_carry ** 2 * scale_factor[-1] * 2)

    # carry = carry.view(512, 512, 4) * scale_factor * 2 / 512
    # carry_loss = torch.mean(carry.sum(dim=-1))

    recon_loss = F.mse_loss(pred_x, target_x)
    
    loss = recon_loss + carry_loss
    
    loss.backward()
    optimizer.step()
    
    return loss.item(), recon_loss.item(), carry_loss.item()

# Initialize models
fake_decoder = binary_decoder(in_features=512, out_features=512, n_bits=4).to(device)
fake_correct_decoder = binary_decoder(in_features=512, out_features=512, n_bits=4).to(device)

# Set up the correct decoder with orthogonal features
with torch.no_grad():
    fake_correct_decoder.weight.zero_()
    for i in range(512):
        fake_correct_decoder.weight[i, i*4] = 1.0

optimizer = optim.Adam(fake_decoder.parameters(), lr=0.001)

# Training settings
n_epochs = 1000000
log_interval = 1 # Print every 100 iterations
best_loss = float('inf')
start_time = time.time()
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Create log file
log_file = f"training_log_{timestamp}.txt"
print(f"Logging to: {log_file}")

with open(log_file, 'w') as f:
    f.write(f"Training started at {datetime.now()}\n")
    f.write(f"Device: {device}\n")
    f.write("Epoch,Loss,Time(s)\n")

try:
    for epoch in range(n_epochs):
        latent_z = generate_binary_latents()
        with torch.no_grad():
            target_x, _ = fake_correct_decoder(latent_z)
        
        loss, recon_loss, carry_loss = train_step(optimizer, latent_z, target_x)
        
        # Log progress
        if (epoch + 1) % log_interval == 0:
            elapsed_time = time.time() - start_time
            print(f"Epoch: {epoch+1}/{n_epochs}, Loss: {loss:.6f}, Recon Loss: {recon_loss:.6f}, Carry Loss: {carry_loss:.6f}, Time: {elapsed_time:.2f}s")
            
            with open(log_file, 'a') as f:
                f.write(f"{epoch+1},{loss:.6f},{recon_loss:.6f},{carry_loss:.6f},{elapsed_time:.2f}\n")
            
            # Save best model
            if loss < best_loss:
                best_loss = loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': fake_decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss,
                }, f'best_decoder_{timestamp}.pt')

except KeyboardInterrupt:
    print("\nTraining interrupted by user")
    # Save final model state
    torch.save({
        'epoch': epoch,
        'model_state_dict': fake_decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, f'interrupted_decoder_{timestamp}.pt')

print("Training completed!")
print(f"Best loss achieved: {best_loss:.6f}")