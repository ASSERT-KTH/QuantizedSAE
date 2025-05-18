import torch
import torch.nn as nn
import torch.nn.functional as F
from SAEs.binary_SAE import BinarySAE
import matplotlib.pyplot as plt
import numpy as np
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os

# Configuration 
config = {
    "input_dim": 512,
    "hidden_dim": 1024,
    "gamma": 4,
    "n_bits": 4,
    "lr": 1e-3,
    "batch_size": 16
}

# Initialize model
model = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"])
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
model.to(device)

# Load a small amount of data
data_path = next(f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt'))
dataset = HiddenStatesTorchDatasetInBinary(os.path.join("dataset/", data_path), config["gamma"], config["n_bits"])

# Just take a small batch for testing
test_batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=config["batch_size"])))
test_batch = test_batch.to(device)

# Store statistics for analysis
stats = {
    'iterations': [],
    'encoder_output_mean': [],
    'encoder_output_std': [],
    'encoder_output_zeros': [],
    'encoder_output_ones': [],
    'weight_mean': [],
    'weight_std': [],
    'weight_max': [],
    'weight_min': [],
    'presigmoid_mean': [],
    'presigmoid_std': [],
    'presigmoid_max': [],
    'presigmoid_min': [],
    'loss': [],
}

# Dict to store gradient info for each iteration
gradient_stats = {
    'iterations': [],
    'mean': [],
    'std': [],
    'max': [],
    'min': [],
}

# Function to collect distribution statistics
def analyze_encoder(model, test_batch, iteration, loss=None):
    # Get access to the linear layer and temperature sigmoid 
    linear_layer = model.encoder[0]
    # temp_sigmoid = model.encoder[1]
    
    # Track linear weights statistics
    weight_stats = {
        'mean': linear_layer.weight.data.mean().item(),
        'std': linear_layer.weight.data.std().item(),
        'max': linear_layer.weight.data.max().item(),
        'min': linear_layer.weight.data.min().item()
    }
    
    # Track pre-sigmoid activations to see if they're exploding
    with torch.no_grad():
        pre_sigmoid = linear_layer(test_batch)
        
    presigmoid_stats = {
        'mean': pre_sigmoid.mean().item(),
        'std': pre_sigmoid.std().item(),
        'max': pre_sigmoid.max().item(),
        'min': pre_sigmoid.min().item()
    }
    
    # Track encoder output distribution
    with torch.no_grad():
        encoder_output = model.encode(test_batch)
        
    # Calculate distribution statistics
    output_mean = encoder_output.mean().item()
    output_std = encoder_output.std().item()
    output_zeros = (encoder_output < 0.01).float().mean().item()
    output_ones = (encoder_output > 0.99).float().mean().item()
    
    # Store statistics
    stats['iterations'].append(iteration)
    stats['encoder_output_mean'].append(output_mean)
    stats['encoder_output_std'].append(output_std)
    stats['encoder_output_zeros'].append(output_zeros)
    stats['encoder_output_ones'].append(output_ones)
    stats['weight_mean'].append(weight_stats['mean'])
    stats['weight_std'].append(weight_stats['std'])
    stats['weight_max'].append(weight_stats['max'])
    stats['weight_min'].append(weight_stats['min'])
    stats['presigmoid_mean'].append(presigmoid_stats['mean'])
    stats['presigmoid_std'].append(presigmoid_stats['std'])
    stats['presigmoid_max'].append(presigmoid_stats['max'])
    stats['presigmoid_min'].append(presigmoid_stats['min'])
    
    if loss is not None:
        stats['loss'].append(loss)
    
    return encoder_output

# Variables to track gradients when needed
current_grad_mean = 0
current_grad_std = 0
current_grad_max = 0 
current_grad_min = 0
collect_grad = False

# Function to register hook for gradient tracking
def setup_gradient_tracking(model):
    linear_layer = model.encoder[0]
    
    def hook_fn(grad):
        global current_grad_mean, current_grad_std, current_grad_max, current_grad_min, collect_grad
        
        if collect_grad:
            current_grad_mean = grad.mean().item()
            current_grad_std = grad.std().item()
            current_grad_max = grad.max().item()
            current_grad_min = grad.min().item()
        
        return grad
    
    handle = linear_layer.weight.register_hook(hook_fn)
    return handle

# Setup optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

# Setup gradient tracking
gradient_hook = setup_gradient_tracking(model)

# Initial analysis
analyze_encoder(model, test_batch, 0)

# Training loop
n_iterations = 10000
binary_latents = []
last_binary_latents_sum = 0
for i in range(1, n_iterations + 1):
    # Forward pass
    binary_latent, recon, carry = model(test_batch)
    binary_latents.append(binary_latent)
    binary_latents_sum = binary_latent.sum(dim=-1).mean()

    if binary_latents_sum == last_binary_latents_sum:
        print("Break break")
    
    # Calculate loss
    scale_factor = (2 ** torch.arange(config["n_bits"], device=device)).float()
    scale_factor = scale_factor / scale_factor.sum()
    
    batch = test_batch.view(config["batch_size"], config["input_dim"], config["n_bits"])
    recon = recon.view(config["batch_size"], config["input_dim"], config["n_bits"])
    
    recon_loss = ((batch - recon) ** 2 * scale_factor).sum(dim=-1).mean()
    sparsity_loss = binary_latent.mean() * 0.01  # Small sparsity penalty
    
    # loss = recon_loss + sparsity_loss
    loss = recon_loss
    
    # Update stats and gradient info between iterations
    if i % 5 == 0:
        # Enable gradient collection for this backward pass
        collect_grad = True
        
        # Backward pass for collecting gradients
        optimizer.zero_grad()
        loss.backward()
        
        # Record gradients
        gradient_stats['iterations'].append(i)
        gradient_stats['mean'].append(current_grad_mean)
        gradient_stats['std'].append(current_grad_std)
        gradient_stats['max'].append(current_grad_max)
        gradient_stats['min'].append(current_grad_min)
        
        # Reset gradient collection flag
        collect_grad = False
        
        # Print progress
        print(f"Iteration {i}, Loss: {loss.item():.6f}")
        analyze_encoder(model, test_batch, i, loss.item())
        
        # Optimizer step after analysis
        optimizer.step()
    else:
        # Regular backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Plot results
plt.figure(figsize=(15, 12))

# Plot encoder output distribution
plt.subplot(3, 2, 1)
plt.plot(stats['iterations'], stats['encoder_output_mean'], label='Mean')
plt.plot(stats['iterations'], stats['encoder_output_std'], label='Std')
plt.title('Encoder Output Statistics')
plt.legend()

plt.subplot(3, 2, 2)
plt.plot(stats['iterations'], stats['encoder_output_zeros'], label='% Near 0')
plt.plot(stats['iterations'], stats['encoder_output_ones'], label='% Near 1')
plt.title('Encoder Saturation')
plt.legend()

# Plot weight statistics
plt.subplot(3, 2, 3)
plt.plot(stats['iterations'], stats['weight_mean'], label='Mean')
plt.plot(stats['iterations'], stats['weight_std'], label='Std')
plt.title('Encoder Weight Statistics')
plt.legend()

plt.subplot(3, 2, 4)
plt.plot(stats['iterations'], stats['weight_max'], label='Max')
plt.plot(stats['iterations'], stats['weight_min'], label='Min')
plt.title('Encoder Weight Range')
plt.legend()

# Plot pre-sigmoid activation statistics
plt.subplot(3, 2, 5)
plt.plot(stats['iterations'], stats['presigmoid_mean'], label='Mean')
plt.plot(stats['iterations'], stats['presigmoid_std'], label='Std')
plt.title('Pre-Sigmoid Activations')
plt.legend()

# Plot gradient statistics
plt.subplot(3, 2, 6)
plt.plot(gradient_stats['iterations'], gradient_stats['mean'], label='Mean')
plt.plot(gradient_stats['iterations'], gradient_stats['std'], label='Std')
plt.title('Gradient Statistics')
plt.legend()

plt.tight_layout()
plt.savefig("encoder_analysis.png")
plt.show()

# Print final statistics
print("\nFinal statistics:")
print(f"Encoder output mean: {stats['encoder_output_mean'][-1]:.4f}")
print(f"Encoder output std: {stats['encoder_output_std'][-1]:.4f}")
print(f"% activations near 0: {stats['encoder_output_zeros'][-1]*100:.2f}%")
print(f"% activations near 1: {stats['encoder_output_ones'][-1]*100:.2f}%")
print(f"Pre-sigmoid range: [{stats['presigmoid_min'][-1]:.2f}, {stats['presigmoid_max'][-1]:.2f}]")
print(f"Weight range: [{stats['weight_min'][-1]:.2f}, {stats['weight_max'][-1]:.2f}]")

# Temperature analysis
print("\nTemperature analysis:")
#temp_value = model.encoder[1].temperature
# print(f"Current sigmoid temperature: {temp_value}")
# if temp_value > 10:
#     print("WARNING: Temperature is extremely high, causing immediate saturation!")
#     print(f"Recommended value: 0.5-2.0, current value: {temp_value}") 