import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from SAEs.binary_SAE_v2 import BinarySAEV2
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os

def test_models():
    """Compare models with and without weight normalization."""
    
    device = 'cpu'
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 1024,  # Smaller for testing
        "gamma": 2
    }
    
    # Load dataset
    dataset_files = [f for f in os.listdir("dataset/") 
                    if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    if not dataset_files:
        print("No dataset files found!")
        return
        
    dataset = HiddenStatesTorchDatasetInBinary(
        os.path.join("dataset/", dataset_files[0]), 
        config["gamma"], 
        config["n_bits"]
    )
    
    # Test original model with weight norm
    print("Testing Binary SAE with weight normalization:")
    model1 = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    test_training(model1, dataset, device, "With WeightNorm")
    
    print("\n" + "="*50 + "\n")
    
    # Test model without weight norm
    print("Testing Binary SAE V2 without weight normalization:")
    model2 = BinarySAEV2(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    test_training(model2, dataset, device, "Without WeightNorm")

def test_training(model, dataset, device, name):
    """Train model for a few steps and monitor stability."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Get initial statistics
    batch = torch.stack([dataset[i] for i in range(256)]).to(device)
    latent, loss = model(batch)
    print(f"Initial loss: {loss.item():.4f}")
    print(f"Initial activation rate: {latent.mean().item():.4f}")
    
    # Train for 20 steps
    losses = []
    activations = []
    
    for step in range(20):
        # Get batch
        indices = torch.randperm(len(dataset))[:256]
        batch = torch.stack([dataset[i] for i in indices]).to(device)
        
        # Forward pass
        latent, loss = model(batch)
        
        # Track metrics
        losses.append(loss.item())
        activations.append(latent.mean().item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if step % 5 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, "
                  f"Act rate={activations[-1]:.4f}, "
                  f"Grad norm={total_norm:.2f}")
    
    # Summary
    print(f"\n{name} Summary:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Final activation rate: {activations[-1]:.4f}")
    print(f"  Loss reduction: {(losses[0] - losses[-1])/losses[0]*100:.1f}%")
    print(f"  Activation stability: std={torch.tensor(activations).std():.4f}")

if __name__ == "__main__":
    test_models() 