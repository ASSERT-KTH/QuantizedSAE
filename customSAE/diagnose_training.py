import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from SAEs.binary_SAE import BinarySAE
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os

def diagnose_binary_sae(model_path=None, config=None):
    """Diagnose issues with Binary SAE training"""
    
    if config is None:
        config = {
            "input_dim": 512,
            "n_bits": 8,
            "hidden_dim": 2048 * 8,
            "gamma": 2,
        }
    
    # Load model
    model = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"])
    if model_path and os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print(f"Loaded model from {model_path}")
    else:
        print("Using randomly initialized model")
    
    model.eval()
    
    # Load a small batch of data
    chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    if chunk_files:
        dataset = HiddenStatesTorchDatasetInBinary(os.path.join("dataset/", chunk_files[0]), config["gamma"], config["n_bits"])
        sample_batch = torch.stack([dataset[i] for i in range(min(32, len(dataset)))])
    else:
        print("No dataset found, using random data")
        sample_batch = torch.randn(32, config["input_dim"] * config["n_bits"])
    
    # Analyze encoder outputs
    with torch.no_grad():
        encoder_out = model.encoder(sample_batch)
        
        # Check activation statistics
        print("\n=== Encoder Output Statistics ===")
        print(f"Mean: {encoder_out.mean():.4f}")
        print(f"Std: {encoder_out.std():.4f}")
        print(f"Min: {encoder_out.min():.4f}")
        print(f"Max: {encoder_out.max():.4f}")
        print(f"% zeros: {(encoder_out == 0).float().mean() * 100:.2f}%")
        
        # Check for dead neurons
        dead_neurons = (encoder_out.abs().max(dim=0)[0] < 1e-6).sum()
        print(f"Dead neurons: {dead_neurons}/{config['hidden_dim']} ({dead_neurons/config['hidden_dim']*100:.2f}%)")
        
        # Analyze binary activations
        latent, recon_loss, polarize_loss = model(sample_batch)
        active_neurons_per_sample = latent.sum(dim=1)
        
        print("\n=== Binary Activation Statistics ===")
        print(f"Average active neurons per sample: {active_neurons_per_sample.mean():.2f}")
        print(f"Std of active neurons: {active_neurons_per_sample.std():.2f}")
        print(f"Min active neurons: {active_neurons_per_sample.min()}")
        print(f"Max active neurons: {active_neurons_per_sample.max()}")
        
        # Check decoder weights
        decoder_weights = model.decoder.weight
        prob_weights = torch.sigmoid(decoder_weights)
        
        print("\n=== Decoder Weight Statistics ===")
        print(f"Mean probability: {prob_weights.mean():.4f}")
        print(f"% near 0 (< 0.1): {(prob_weights < 0.1).float().mean() * 100:.2f}%")
        print(f"% near 1 (> 0.9): {(prob_weights > 0.9).float().mean() * 100:.2f}%")
        print(f"% in middle (0.3-0.7): {((prob_weights > 0.3) & (prob_weights < 0.7)).float().mean() * 100:.2f}%")
        
        # Visualize distributions
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Encoder output distribution
        axes[0, 0].hist(encoder_out.flatten().numpy(), bins=50, alpha=0.7)
        axes[0, 0].set_title('Encoder Output Distribution')
        axes[0, 0].set_xlabel('Activation Value')
        axes[0, 0].set_ylabel('Count')
        
        # Active neurons per sample
        axes[0, 1].hist(active_neurons_per_sample.numpy(), bins=30, alpha=0.7)
        axes[0, 1].set_title('Active Neurons per Sample')
        axes[0, 1].set_xlabel('Number of Active Neurons')
        axes[0, 1].set_ylabel('Count')
        
        # Decoder weight probabilities
        axes[1, 0].hist(prob_weights.flatten().numpy(), bins=50, alpha=0.7)
        axes[1, 0].set_title('Decoder Weight Probabilities')
        axes[1, 0].set_xlabel('Sigmoid(weight)')
        axes[1, 0].set_ylabel('Count')
        
        # Gradient flow analysis
        sample_batch.requires_grad = True
        latent, recon_loss, polarize_loss = model(sample_batch)
        loss = recon_loss + 0.01 * polarize_loss
        loss.backward()
        
        encoder_grad_norm = model.encoder[0].weight.grad.norm().item()
        decoder_grad_norm = model.decoder.weight.grad.norm().item()
        
        axes[1, 1].bar(['Encoder', 'Decoder'], [encoder_grad_norm, decoder_grad_norm])
        axes[1, 1].set_title('Gradient Norms')
        axes[1, 1].set_ylabel('Gradient Norm')
        
        plt.tight_layout()
        plt.savefig('binary_sae_diagnostics.png')
        print("\nDiagnostic plots saved to 'binary_sae_diagnostics.png'")
        
        print("\n=== Gradient Flow ===")
        print(f"Encoder gradient norm: {encoder_grad_norm:.6f}")
        print(f"Decoder gradient norm: {decoder_grad_norm:.6f}")
        
        if encoder_grad_norm < 1e-6:
            print("WARNING: Encoder gradients are vanishing!")
        if decoder_grad_norm < 1e-6:
            print("WARNING: Decoder gradients are vanishing!")
            
        print("\n=== Loss Values ===")
        print(f"Reconstruction loss: {recon_loss.item():.6f}")
        print(f"Polarization loss: {polarize_loss.item():.6f}")

if __name__ == "__main__":
    # You can specify a model path if you have a saved model
    diagnose_binary_sae(model_path="SAEs/b_sae_16384.pth")