import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from SAEs.binary_SAE_improved_post_quant import BinarySAEImprovedPostQuant
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os
import matplotlib.pyplot as plt

def analyze_weight_distribution(model, name):
    """Analyze decoder weight distribution."""
    with torch.no_grad():
        if hasattr(model.decoder, 'weight'):
            prob_weights = torch.sigmoid(model.decoder.weight).flatten()
        else:
            return None
        
        # Statistics
        close_to_0 = (prob_weights < 0.2).float().mean()
        close_to_1 = (prob_weights > 0.8).float().mean()
        close_to_half = ((prob_weights > 0.4) & (prob_weights < 0.6)).float().mean()
        
        print(f"\n{name} weight distribution:")
        print(f"  Weights < 0.2 (close to 0): {close_to_0:.1%}")
        print(f"  Weights > 0.8 (close to 1): {close_to_1:.1%}")
        print(f"  Weights 0.4-0.6 (close to 0.5): {close_to_half:.1%}")
        print(f"  Mean: {prob_weights.mean():.4f}")
        print(f"  Std: {prob_weights.std():.4f}")
        
        return {
            'prob_weights': prob_weights,
            'close_to_0': close_to_0.item(),
            'close_to_1': close_to_1.item(),
            'close_to_half': close_to_half.item()
        }

def test_quantization_impact(model, dataset, device='cpu'):
    """Test quantization error."""
    model.eval()
    
    # Get test batch
    batch = torch.stack([dataset[i] for i in range(100)]).to(device)
    
    with torch.no_grad():
        # Continuous version
        if isinstance(model, BinarySAEImprovedPostQuant):
            # New regularized model
            latent, total_loss, recon_loss, reg_loss = model(batch)
            continuous_loss = recon_loss.item()
        else:
            # Original model
            latent, continuous_loss = model(batch)
            continuous_loss = continuous_loss.item()
        
        # Quantize decoder weights
        if hasattr(model.decoder, 'weight'):
            original_weights = model.decoder.weight.data.clone()
            
            # Quantize
            prob_weights = torch.sigmoid(model.decoder.weight)
            binary_weights = (prob_weights > 0.5).float()
            binary_logits = torch.log(binary_weights.clamp(1e-7, 1-1e-7) / 
                                    (1 - binary_weights.clamp(1e-7, 1-1e-7)))
            
            model.decoder.weight.data = binary_logits
            
            # Test quantized
            if isinstance(model, BinarySAEImprovedPostQuant):
                _, total_loss, quantized_loss, _ = model(batch)
                quantized_loss = quantized_loss.item()
            else:
                _, quantized_loss = model(batch)
                quantized_loss = quantized_loss.item()
            
            # Restore
            model.decoder.weight.data = original_weights
            
            error = abs(quantized_loss - continuous_loss) / continuous_loss
            
            print(f"  Continuous loss: {continuous_loss:.6f}")
            print(f"  Quantized loss: {quantized_loss:.6f}")
            print(f"  Relative error: {error:.2%}")
            
            return error
    
    return 0

def compare_approaches():
    """Compare original vs regularized continuous decoder."""
    
    device = 'cpu'
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 1024,  # Smaller for faster testing
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
    
    print("="*70)
    print("COMPARING CONTINUOUS DECODER APPROACHES")
    print("="*70)
    
    # Test 1: Original continuous decoder
    print("\n1. Original Continuous Decoder (no regularization)")
    print("-" * 50)
    
    model1 = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    train_model(model1, dataset, device, steps=100, reg_weight=0)
    weight_stats1 = analyze_weight_distribution(model1, "Original")
    quant_error1 = test_quantization_impact(model1, dataset, device)
    
    # Test 2: Regularized continuous decoder
    print("\n2. Regularized Continuous Decoder")
    print("-" * 50)
    
    model2 = BinarySAEImprovedPostQuant(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    train_model(model2, dataset, device, steps=100, reg_weight=1e-2)
    weight_stats2 = analyze_weight_distribution(model2, "Regularized")
    quant_error2 = test_quantization_impact(model2, dataset, device)
    
    # Comparison
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    print(f"{'Metric':<30} | {'Original':<15} | {'Regularized':<15}")
    print("-" * 65)
    print(f"{'Weights close to 0.5':<30} | {weight_stats1['close_to_half']:<15.1%} | {weight_stats2['close_to_half']:<15.1%}")
    print(f"{'Weights close to 0 or 1':<30} | {weight_stats1['close_to_0'] + weight_stats1['close_to_1']:<15.1%} | {weight_stats2['close_to_0'] + weight_stats2['close_to_1']:<15.1%}")
    print(f"{'Quantization error':<30} | {quant_error1:<15.1%} | {quant_error2:<15.1%}")
    
    # Verdict
    if quant_error2 < quant_error1 * 0.1:
        print("\n✓ Regularization SIGNIFICANTLY improves quantization!")
    elif quant_error2 < quant_error1 * 0.5:
        print("\n⚠ Regularization moderately improves quantization")
    else:
        print("\n✗ Regularization doesn't help much")
    
    return model1, model2

def train_model(model, dataset, device, steps=100, reg_weight=1e-3):
    """Train model for a few steps."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    for step in range(steps):
        # Get batch
        indices = torch.randperm(len(dataset))[:256]
        batch = torch.stack([dataset[i] for i in indices]).to(device)
        
        # Forward pass
        if isinstance(model, BinarySAEImprovedPostQuant):
            # Regularized model
            latent, total_loss, recon_loss, reg_loss = model(batch, reg_weight=reg_weight)
            loss = total_loss
        else:
            # Original model
            latent, loss = model(batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if step % 25 == 0:
            act_rate = latent.mean().item()
            if isinstance(model, BinarySAEImprovedPostQuant):
                print(f"  Step {step}: Total={total_loss.item():.4f}, Recon={recon_loss.item():.4f}, Reg={reg_loss.item():.6f}, Act={act_rate:.4f}")
            else:
                print(f"  Step {step}: Loss={loss.item():.4f}, Act rate={act_rate:.4f}")

if __name__ == "__main__":
    compare_approaches() 