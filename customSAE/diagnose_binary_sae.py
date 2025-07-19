import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from SAEs.binary_SAE_fixed import BinarySAEFixed
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import numpy as np
import matplotlib.pyplot as plt
import os

def diagnose_model(model, dataset, device='cuda', n_samples=100):
    """Diagnose what's happening with the binary SAE."""
    
    model.eval()
    model = model.to(device)
    
    # Get a batch of data
    batch_indices = np.random.choice(len(dataset), n_samples, replace=False)
    batch = torch.stack([dataset[i] for i in batch_indices]).to(device)
    
    with torch.no_grad():
        # Forward pass
        latent, loss = model(batch)
        
        # Analyze latent activations
        activation_rate = latent.mean().item()
        activated_per_sample = latent.sum(dim=-1)
        
        # Get decoder weights statistics
        if hasattr(model.decoder, 'weight'):
            decoder_weights = torch.sigmoid(model.decoder.weight)
            hard_weights = (decoder_weights > 0.5).float()
            weight_activation_rate = hard_weights.mean().item()
        
        # Reconstruct integer values
        powers = 2 ** torch.arange(model.n_bits, device=device).float()
        powers[-1] *= -1  # Signed representation
        
        # Convert input binary to integers
        input_ints = (batch.view(batch.shape[0], -1, model.n_bits) * powers).sum(-1)
        
        # Get predictions (manually compute what decoder should output)
        if hasattr(model.decoder, 'weight'):
            int_weights = (hard_weights.view(model.decoder.in_features, -1, model.n_bits) * powers).sum(-1)
            pred_ints = (latent.unsqueeze(-1) * int_weights.unsqueeze(0)).sum(-2)
        
    print("=== Binary SAE Diagnostic Report ===")
    print(f"Loss: {loss.item():.4f}")
    print(f"Latent activation rate: {activation_rate:.4f}")
    print(f"Activated neurons per sample: {activated_per_sample.mean():.1f} ± {activated_per_sample.std():.1f}")
    print(f"Decoder weight activation rate: {weight_activation_rate:.4f}")
    
    print(f"\nInput integer statistics:")
    print(f"  Mean: {input_ints.mean():.2f}")
    print(f"  Std: {input_ints.std():.2f}")
    print(f"  Min: {input_ints.min():.2f}")
    print(f"  Max: {input_ints.max():.2f}")
    
    if hasattr(model.decoder, 'weight'):
        print(f"\nPrediction integer statistics:")
        print(f"  Mean: {pred_ints.mean():.2f}")
        print(f"  Std: {pred_ints.std():.2f}")
        print(f"  Min: {pred_ints.min():.2f}")
        print(f"  Max: {pred_ints.max():.2f}")
        
        # Check if predictions are all zeros
        if torch.allclose(pred_ints, torch.zeros_like(pred_ints)):
            print("\n⚠️  WARNING: Model is outputting all zeros!")
        
        # Analyze reconstruction error
        recon_error = (pred_ints - input_ints).abs()
        print(f"\nReconstruction error statistics:")
        print(f"  Mean absolute error: {recon_error.mean():.2f}")
        print(f"  Max absolute error: {recon_error.max():.2f}")
    
    # Plot histogram of activations
    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.hist(activated_per_sample.cpu().numpy(), bins=50)
    plt.xlabel('Number of activated neurons')
    plt.ylabel('Frequency')
    plt.title('Distribution of Activated Neurons per Sample')
    
    plt.subplot(1, 2, 2)
    plt.hist(input_ints.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Input')
    if hasattr(model.decoder, 'weight'):
        plt.hist(pred_ints.cpu().numpy().flatten(), bins=50, alpha=0.5, label='Prediction')
    plt.xlabel('Integer value')
    plt.ylabel('Frequency')
    plt.title('Distribution of Integer Values')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('binary_sae_diagnostic.png')
    print("\nDiagnostic plots saved to 'binary_sae_diagnostic.png'")
    
    return {
        'loss': loss.item(),
        'activation_rate': activation_rate,
        'activated_per_sample': activated_per_sample.cpu(),
        'input_ints': input_ints.cpu(),
        'pred_ints': pred_ints.cpu() if hasattr(model.decoder, 'weight') else None
    }


if __name__ == "__main__":
    # Test with your current model
    print("Testing original Binary SAE...")
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 2048 * 8,
        "gamma": 2
    }
    
    # Load a sample dataset
    dataset_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    if dataset_files:
        dataset = HiddenStatesTorchDatasetInBinary(
            os.path.join("dataset/", dataset_files[0]), 
            config["gamma"], 
            config["n_bits"]
        )
        
        # Test original model
        model_orig = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"])
        if os.path.exists("SAEs/b_sae_16384.pth"):
            model_orig.load_state_dict(torch.load("SAEs/b_sae_16384.pth"))
            print("Loaded existing model weights")
        
        diagnose_model(model_orig, dataset)
        
        print("\n" + "="*50 + "\n")
        
        # Test fixed model
        print("Testing fixed Binary SAE...")
        model_fixed = BinarySAEFixed(
            config["input_dim"], 
            config["hidden_dim"], 
            config["n_bits"],
            config["gamma"]
        )
        diagnose_model(model_fixed, dataset) 