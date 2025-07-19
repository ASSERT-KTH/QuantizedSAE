import torch
import torch.nn as nn
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os
import matplotlib.pyplot as plt

class TestBinarySAEWithBias(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_bits, bias_init):
        super().__init__()
        self.n_bits = n_bits
        
        # Encoder without weight norm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_bits, hidden_dim),
            nn.Sigmoid()
        )
        
        # Xavier initialization for weights
        nn.init.xavier_uniform_(self.encoder[0].weight)
        
        # Custom bias initialization
        nn.init.constant_(self.encoder[0].bias, bias_init)
        
        # Simple decoder for testing
        self.decoder_weight = nn.Parameter(torch.randn(hidden_dim, input_dim * n_bits) * 0.01)
    
    def forward(self, x):
        latent = self.encoder(x)
        binary_latent = (latent > 0.5).float()
        latent_ste = latent + (binary_latent - latent.detach())
        
        # Simple reconstruction
        recon = torch.matmul(latent_ste, self.decoder_weight)
        loss = nn.functional.mse_loss(recon, x)
        
        return latent_ste, loss

def test_bias_initializations():
    """Test how bias initialization affects activation collapse."""
    
    device = 'cpu'
    hidden_dim = 16384
    input_dim = 512
    n_bits = 8
    
    # Load dataset
    dataset_files = [f for f in os.listdir("dataset/") 
                    if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    if not dataset_files:
        print("No dataset files found!")
        return
        
    dataset = HiddenStatesTorchDatasetInBinary(
        os.path.join("dataset/", dataset_files[0]), 
        gamma=2, 
        n_bits=n_bits
    )
    
    # Test different bias initializations
    # sigmoid(x) = 0.05 when x ≈ -2.94
    # sigmoid(x) = 0.10 when x ≈ -2.20
    # sigmoid(x) = 0.50 when x = 0
    bias_inits = [
        ("Zero (50% activation)", 0.0),
        ("Negative small (-1.0)", -1.0),
        ("Target 10% activation (-2.2)", -2.2),
        ("Target 5% activation (-2.94)", -2.94),
        ("Target 1% activation (-4.6)", -4.6)
    ]
    
    results = {}
    
    for name, bias_init in bias_inits:
        print(f"\n{'='*50}")
        print(f"Testing: {name}, bias={bias_init:.2f}")
        print(f"Expected initial activation: {torch.sigmoid(torch.tensor(bias_init)):.3f}")
        print(f"{'='*50}")
        
        # Create model
        model = TestBinarySAEWithBias(input_dim, hidden_dim, n_bits, bias_init).to(device)
        
        # Use lower learning rate for stability
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        
        # Track metrics
        losses = []
        activations = []
        
        # Get initial activation rate
        with torch.no_grad():
            indices = torch.randperm(len(dataset))[:256]
            batch = torch.stack([dataset[i] for i in indices]).to(device)
            latent, _ = model(batch)
            init_act_rate = latent.mean().item()
            print(f"Actual initial activation rate: {init_act_rate:.3f}")
        
        # Train for 50 steps
        for step in range(50):
            # Get batch
            indices = torch.randperm(len(dataset))[:256]
            batch = torch.stack([dataset[i] for i in indices]).to(device)
            
            # Forward pass
            latent, loss = model(batch)
            
            # Track metrics
            act_rate = latent.mean().item()
            losses.append(loss.item())
            activations.append(act_rate)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Aggressive gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            optimizer.step()
            
            if step % 10 == 0:
                print(f"Step {step}: Loss={loss.item():.4f}, Act rate={act_rate:.4f}")
        
        results[name] = {
            'losses': losses,
            'activations': activations,
            'init_act_rate': init_act_rate,
            'final_act_rate': activations[-1],
            'collapsed': activations[-1] < 0.02
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = plt.cm.viridis(torch.linspace(0, 1, len(bias_inits)))
    
    for (name, data), color in zip(results.items(), colors):
        ax1.plot(data['losses'], label=name, color=color)
        ax2.plot(data['activations'], label=name, color=color)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend(fontsize=8)
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Activation Rate')
    ax2.set_title('Mean Activation Rate Over Training')
    ax2.legend(fontsize=8)
    ax2.set_ylim([0, 0.6])
    ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.5, label='5% target')
    
    plt.tight_layout()
    plt.savefig('bias_initialization_comparison.png')
    print("\nResults saved to bias_initialization_comparison.png")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY:")
    print("="*60)
    print(f"{'Configuration':<30} | {'Initial':<8} | {'Final':<8} | {'Status':<10}")
    print("-"*60)
    for name, data in results.items():
        status = "COLLAPSED" if data['collapsed'] else "STABLE"
        print(f"{name:<30} | {data['init_act_rate']:<8.3f} | {data['final_act_rate']:<8.3f} | {status:<10}")

if __name__ == "__main__":
    test_bias_initializations() 