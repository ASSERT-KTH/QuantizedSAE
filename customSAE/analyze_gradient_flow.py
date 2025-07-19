import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import matplotlib.pyplot as plt
import os

def analyze_gradient_flow(model, dataset, device='cpu', n_steps=10):
    """Analyze gradient flow through the binary SAE."""
    
    model = model.to(device)
    model.train()
    
    # Track gradients over multiple steps
    encoder_grads = []
    decoder_grads = []
    losses = []
    activations = []
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(n_steps):
        # Get a batch
        batch_indices = torch.randperm(len(dataset))[:512]
        batch = torch.stack([dataset[i] for i in batch_indices]).to(device)
        
        # Forward pass with gradient tracking
        latent, loss = model(batch)
        
        # Track activation statistics
        with torch.no_grad():
            activation_rate = latent.mean().item()
            activations.append(activation_rate)
            
            # Check if activations are collapsing
            if activation_rate < 0.01:
                print(f"⚠️  Step {step}: Activations collapsed to near zero ({activation_rate:.4f})")
            elif activation_rate > 0.99:
                print(f"⚠️  Step {step}: Activations saturated to near one ({activation_rate:.4f})")
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Analyze gradients
        with torch.no_grad():
            # Calculate gradient norms properly for weight-normed layers
            enc_grad_norm = 0
            dec_grad_norm = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if 'encoder' in name:
                        enc_grad_norm += param.grad.norm().item() ** 2
                    elif 'decoder' in name:
                        dec_grad_norm += param.grad.norm().item() ** 2
            enc_grad_norm = enc_grad_norm ** 0.5
            dec_grad_norm = dec_grad_norm ** 0.5
            
            encoder_grads.append(enc_grad_norm)
            decoder_grads.append(dec_grad_norm)
            
            # Check for vanishing/exploding gradients
            if enc_grad_norm < 1e-6:
                print(f"⚠️  Step {step}: Encoder gradients vanishing ({enc_grad_norm:.2e})")
            elif enc_grad_norm > 1e3:
                print(f"⚠️  Step {step}: Encoder gradients exploding ({enc_grad_norm:.2e})")
                
            if dec_grad_norm < 1e-6:
                print(f"⚠️  Step {step}: Decoder gradients vanishing ({dec_grad_norm:.2e})")
            elif dec_grad_norm > 1e3:
                print(f"⚠️  Step {step}: Decoder gradients exploding ({dec_grad_norm:.2e})")
        
        losses.append(loss.item())
        optimizer.step()
        
        print(f"Step {step}: Loss={loss.item():.4f}, Enc grad={enc_grad_norm:.4f}, Dec grad={dec_grad_norm:.4f}, Act rate={activation_rate:.4f}")
    
    # Plot diagnostics
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    axes[0, 0].plot(losses)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')
    axes[0, 0].set_yscale('log')
    
    # Gradient norms
    axes[0, 1].plot(encoder_grads, label='Encoder')
    axes[0, 1].plot(decoder_grads, label='Decoder')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Gradient Norm')
    axes[0, 1].set_title('Gradient Norms')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    
    # Activation rate
    axes[1, 0].plot(activations)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Activation Rate')
    axes[1, 0].set_title('Mean Activation Rate')
    axes[1, 0].set_ylim([0, 1])
    
    # Gradient ratio
    grad_ratio = [e/d if d > 0 else 0 for e, d in zip(encoder_grads, decoder_grads)]
    axes[1, 1].plot(grad_ratio)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('Encoder/Decoder Gradient Ratio')
    axes[1, 1].set_title('Gradient Balance')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    plt.savefig('gradient_flow_analysis.png')
    print("\nGradient flow analysis saved to 'gradient_flow_analysis.png'")
    
    # Analyze weight distributions
    with torch.no_grad():
        # Encoder weights
        enc_weights = model.encoder[0].weight
        print(f"\nEncoder weight statistics:")
        print(f"  Mean: {enc_weights.mean():.4f}")
        print(f"  Std: {enc_weights.std():.4f}")
        print(f"  Min: {enc_weights.min():.4f}")
        print(f"  Max: {enc_weights.max():.4f}")
        
        # Decoder weights (before sigmoid)
        dec_weights = model.decoder.weight
        print(f"\nDecoder weight statistics (pre-sigmoid):")
        print(f"  Mean: {dec_weights.mean():.4f}")
        print(f"  Std: {dec_weights.std():.4f}")
        print(f"  Min: {dec_weights.min():.4f}")
        print(f"  Max: {dec_weights.max():.4f}")
        
        # Decoder weights (after sigmoid)
        dec_weights_sigmoid = torch.sigmoid(dec_weights)
        hard_weights = (dec_weights_sigmoid > 0.5).float()
        print(f"\nDecoder binary weight statistics:")
        print(f"  Fraction of 1s: {hard_weights.mean():.4f}")
        print(f"  Total 1s: {hard_weights.sum().item():.0f} / {hard_weights.numel()}")

def main():
    # Configuration matching your setup
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 2048 * 8,
        "gamma": 2
    }
    
    # Load dataset
    dataset_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    if not dataset_files:
        print("No dataset files found!")
        return
        
    dataset = HiddenStatesTorchDatasetInBinary(
        os.path.join("dataset/", dataset_files[0]), 
        config["gamma"], 
        config["n_bits"]
    )
    
    # Create model
    model = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"])
    
    print("Analyzing gradient flow in Binary SAE...")
    analyze_gradient_flow(model, dataset, n_steps=20)

if __name__ == "__main__":
    main() 