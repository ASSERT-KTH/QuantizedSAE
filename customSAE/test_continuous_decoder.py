import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os
import matplotlib.pyplot as plt
import numpy as np

def analyze_decoder_weights(model):
    """Analyze the distribution of decoder weights and quantization errors."""
    
    with torch.no_grad():
        # Get decoder weights after sigmoid
        prob_weights = torch.sigmoid(model.decoder.weight)
        
        # Analyze distribution
        weights_flat = prob_weights.flatten()
        
        # Calculate distances to nearest binary values
        dist_to_zero = torch.abs(weights_flat - 0.0)
        dist_to_one = torch.abs(weights_flat - 1.0)
        min_dist_to_binary = torch.minimum(dist_to_zero, dist_to_one)
        
        # Weights close to 0.5 (worst case for quantization)
        close_to_half = torch.abs(weights_flat - 0.5) < 0.1  # Within 0.1 of 0.5
        
        print(f"\nDecoder Weight Analysis:")
        print(f"  Mean weight value: {weights_flat.mean():.4f}")
        print(f"  Std of weights: {weights_flat.std():.4f}")
        print(f"  Weights < 0.2: {(weights_flat < 0.2).float().mean():.1%}")
        print(f"  Weights > 0.8: {(weights_flat > 0.8).float().mean():.1%}")
        print(f"  Weights close to 0.5 (±0.1): {close_to_half.float().mean():.1%}")
        print(f"  Mean distance to nearest binary: {min_dist_to_binary.mean():.4f}")
        print(f"  Max distance to nearest binary: {min_dist_to_binary.max():.4f}")
        
        return {
            'prob_weights': prob_weights,
            'min_dist_to_binary': min_dist_to_binary,
            'close_to_half_fraction': close_to_half.float().mean().item()
        }

def test_quantization_error(model, dataset, device='cpu', n_samples=100):
    """Test the impact of post-training quantization."""
    
    model.eval()
    model = model.to(device)
    
    # Get test batch
    batch_indices = torch.randperm(len(dataset))[:n_samples]
    batch = torch.stack([dataset[i] for i in batch_indices]).to(device)
    
    with torch.no_grad():
        # Forward pass with continuous decoder weights
        latent, loss_continuous = model(batch)
        
        # Now simulate post-training quantization
        original_weight = model.decoder.weight.data.clone()
        
        # Quantize decoder weights
        prob_weights = torch.sigmoid(model.decoder.weight)
        quantized_weights = (prob_weights > 0.5).float()
        
        # Convert back to logits (inverse sigmoid)
        # For numerical stability, clamp to avoid infinite values
        quantized_logits = torch.log(quantized_weights.clamp(1e-7, 1-1e-7) / 
                                   (1 - quantized_weights.clamp(1e-7, 1-1e-7)))
        
        # Replace decoder weights with quantized version
        model.decoder.weight.data = quantized_logits
        
        # Forward pass with quantized decoder weights
        _, loss_quantized = model(batch)
        
        # Restore original weights
        model.decoder.weight.data = original_weight
    
    error = abs(loss_quantized.item() - loss_continuous.item())
    relative_error = error / loss_continuous.item()
    
    print(f"\nQuantization Error Analysis:")
    print(f"  Continuous decoder loss: {loss_continuous.item():.6f}")
    print(f"  Quantized decoder loss: {loss_quantized.item():.6f}")
    print(f"  Absolute error: {error:.6f}")
    print(f"  Relative error: {relative_error:.2%}")
    
    return {
        'continuous_loss': loss_continuous.item(),
        'quantized_loss': loss_quantized.item(),
        'absolute_error': error,
        'relative_error': relative_error
    }

def train_and_analyze_continuous_decoder():
    """Train the model with continuous decoder and analyze weight evolution."""
    
    device = 'cpu'
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 2048,
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
    
    # Create model
    model = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Track metrics during training
    losses = []
    activation_rates = []
    weight_analyses = []
    
    print("Training Binary SAE with continuous decoder...")
    print("="*60)
    
    for step in range(200):
        # Get batch
        indices = torch.randperm(len(dataset))[:256]
        batch = torch.stack([dataset[i] for i in indices]).to(device)
        
        # Forward pass
        latent, loss = model(batch)
        
        # Track metrics
        losses.append(loss.item())
        activation_rates.append(latent.mean().item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Analyze weights periodically
        if step % 50 == 0:
            print(f"\nStep {step}:")
            print(f"  Loss: {loss.item():.4f}")
            print(f"  Activation rate: {activation_rates[-1]:.4f}")
            
            weight_analysis = analyze_decoder_weights(model)
            weight_analyses.append(weight_analysis)
            
            # Test quantization error
            quant_analysis = test_quantization_error(model, dataset, device)
    
    # Final analysis
    print("\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)
    
    final_weight_analysis = analyze_decoder_weights(model)
    final_quant_analysis = test_quantization_error(model, dataset, device)
    
    # Plot results
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curve
    ax1.plot(losses)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.set_yscale('log')
    
    # Activation rate
    ax2.plot(activation_rates)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Activation Rate')
    ax2.set_title('Activation Rate During Training')
    ax2.set_ylim([0, max(activation_rates) * 1.1])
    
    # Weight distribution histogram
    with torch.no_grad():
        prob_weights = torch.sigmoid(model.decoder.weight).flatten()
        ax3.hist(prob_weights.cpu().numpy(), bins=50, alpha=0.7)
        ax3.axvline(x=0.5, color='red', linestyle='--', label='0.5 (worst for quantization)')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Final Decoder Weight Distribution')
        ax3.legend()
    
    # Distance to binary values
    dist_to_zero = torch.abs(prob_weights - 0.0)
    dist_to_one = torch.abs(prob_weights - 1.0)
    min_dist = torch.minimum(dist_to_zero, dist_to_one)
    
    ax4.hist(min_dist.cpu().numpy(), bins=50, alpha=0.7)
    ax4.set_xlabel('Distance to Nearest Binary Value')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Quantization Error Distribution')
    
    plt.tight_layout()
    plt.savefig('continuous_decoder_analysis.png')
    print("\nAnalysis plots saved to continuous_decoder_analysis.png")
    
    # Summary
    print(f"\nSUMMARY:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Final activation rate: {activation_rates[-1]:.4f}")
    print(f"  Activation collapsed: {'YES' if activation_rates[-1] < 0.01 else 'NO'}")
    print(f"  Weights close to 0.5: {final_weight_analysis['close_to_half_fraction']:.1%}")
    print(f"  Quantization relative error: {final_quant_analysis['relative_error']:.2%}")
    
    # Verdict
    if final_quant_analysis['relative_error'] < 0.1:  # Less than 10% error
        print("\n✓ Post-quantization approach looks PROMISING!")
    elif final_quant_analysis['relative_error'] < 0.5:  # Less than 50% error
        print("\n⚠ Post-quantization approach is MODERATE - might work")
    else:
        print("\n✗ Post-quantization approach has HIGH error - problematic")

if __name__ == "__main__":
    train_and_analyze_continuous_decoder() 