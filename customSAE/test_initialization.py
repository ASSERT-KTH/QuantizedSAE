import torch
import torch.nn as nn
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os
import matplotlib.pyplot as plt

class TestBinarySAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_bits, init_std):
        super().__init__()
        self.n_bits = n_bits
        
        # Encoder without weight norm
        self.encoder = nn.Sequential(
            nn.Linear(input_dim * n_bits, hidden_dim),
            nn.Sigmoid()
        )
        
        # Custom initialization
        nn.init.normal_(self.encoder[0].weight, std=init_std)
        nn.init.zeros_(self.encoder[0].bias)
        
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

def test_different_initializations():
    """Test how initialization affects activation collapse."""
    
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
    
    # Test different initialization scales
    init_stds = [
        ("Original (1/√hidden_dim)", 1/torch.sqrt(torch.tensor(hidden_dim)).item()),
        ("Xavier (√(2/(in+out)))", torch.sqrt(torch.tensor(2.0 / (input_dim * n_bits + hidden_dim))).item()),
        ("Larger (0.01)", 0.01),
        ("Much larger (0.1)", 0.1)
    ]
    
    results = {}
    
    for name, init_std in init_stds:
        print(f"\n{'='*50}")
        print(f"Testing: {name}, std={init_std:.6f}")
        print(f"{'='*50}")
        
        # Create model
        model = TestBinarySAE(input_dim, hidden_dim, n_bits, init_std).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        
        # Track metrics
        losses = []
        activations = []
        
        # Train for 20 steps
        for step in range(20):
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
                print(f"Step {step}: Loss={loss.item():.4f}, Act rate={act_rate:.4f}, Grad norm={total_norm:.2f}")
        
        results[name] = {
            'losses': losses,
            'activations': activations,
            'final_act_rate': activations[-1],
            'collapsed': activations[-1] < 0.05
        }
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    for name, data in results.items():
        ax1.plot(data['losses'], label=name)
        ax2.plot(data['activations'], label=name)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.set_yscale('log')
    
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Activation Rate')
    ax2.set_title('Mean Activation Rate')
    ax2.legend()
    ax2.set_ylim([0, 0.6])
    
    plt.tight_layout()
    plt.savefig('initialization_comparison.png')
    print("\nResults saved to initialization_comparison.png")
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY:")
    print("="*50)
    for name, data in results.items():
        status = "COLLAPSED" if data['collapsed'] else "OK"
        print(f"{name}: Final act rate = {data['final_act_rate']:.4f} [{status}]")

if __name__ == "__main__":
    test_different_initializations() 