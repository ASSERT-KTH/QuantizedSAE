import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
from SAEs.binary_SAE_topk import TopKBinarySAE
from hidden_state_dataset import HiddenStatesTorchDatasetInBinary
import os
import matplotlib.pyplot as plt

def compare_models():
    """Compare sigmoid-based vs TopK-based binary SAE."""
    
    device = 'cpu'
    config = {
        "input_dim": 512,
        "n_bits": 8,
        "hidden_dim": 2048,  # Smaller for faster testing
        "gamma": 2,
        "k": 100  # 5% of 2048
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
    
    # Model 1: Original Binary SAE with sigmoid
    print("="*60)
    print("Testing Sigmoid-based Binary SAE")
    print("="*60)
    
    model_sigmoid = BinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"]).to(device)
    results_sigmoid = train_and_evaluate(model_sigmoid, dataset, device, "Sigmoid")
    
    print("\n" + "="*60)
    print("Testing TopK-based Binary SAE")
    print("="*60)
    
    # Model 2: TopK Binary SAE
    model_topk = TopKBinarySAE(config["input_dim"], config["hidden_dim"], config["n_bits"], k=config["k"]).to(device)
    results_topk = train_and_evaluate(model_topk, dataset, device, "TopK")
    
    # Plot comparison
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax1.plot(results_sigmoid['losses'], label='Sigmoid', color='red')
    ax1.plot(results_topk['losses'], label='TopK', color='blue')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Comparison')
    ax1.legend()
    ax1.set_yscale('log')
    
    # Activation rates
    ax2.plot(results_sigmoid['activations'], label='Sigmoid', color='red')
    ax2.axhline(y=results_topk['activations'][0], color='blue', linestyle='--', label='TopK (constant)')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Activation Rate')
    ax2.set_title('Activation Rate Over Time')
    ax2.legend()
    ax2.set_ylim([0, 0.6])
    
    # Gradient norms
    ax3.plot(results_sigmoid['grad_norms'], label='Sigmoid', color='red')
    ax3.plot(results_topk['grad_norms'], label='TopK', color='blue')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norm Evolution')
    ax3.legend()
    ax3.set_yscale('log')
    
    # Final comparison bar chart
    categories = ['Final Loss', 'Final Act Rate', 'Min Act Rate']
    sigmoid_vals = [
        results_sigmoid['losses'][-1],
        results_sigmoid['activations'][-1],
        min(results_sigmoid['activations'])
    ]
    topk_vals = [
        results_topk['losses'][-1],
        results_topk['activations'][-1],
        min(results_topk['activations'])
    ]
    
    x = range(len(categories))
    width = 0.35
    ax4.bar([i - width/2 for i in x], sigmoid_vals, width, label='Sigmoid', color='red', alpha=0.7)
    ax4.bar([i + width/2 for i in x], topk_vals, width, label='TopK', color='blue', alpha=0.7)
    ax4.set_ylabel('Value')
    ax4.set_title('Final Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('sigmoid_vs_topk_comparison.png')
    print("\nComparison plot saved to sigmoid_vs_topk_comparison.png")

def train_and_evaluate(model, dataset, device, name):
    """Train model and track metrics."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    losses = []
    activations = []
    grad_norms = []
    
    # Train for 100 steps
    for step in range(100):
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
        
        # Calculate gradient norm
        total_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        grad_norms.append(total_norm)
        
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        if step % 20 == 0:
            activated_neurons = latent.sum(dim=-1).mean().item()
            print(f"Step {step}: Loss={loss.item():.4f}, "
                  f"Act rate={act_rate:.4f}, "
                  f"Activated neurons={activated_neurons:.1f}, "
                  f"Grad norm={total_norm:.2f}")
    
    # Summary
    print(f"\n{name} Summary:")
    print(f"  Final loss: {losses[-1]:.4f}")
    print(f"  Final activation rate: {activations[-1]:.4f}")
    print(f"  Min activation rate: {min(activations):.4f}")
    print(f"  Activation collapsed: {'YES' if min(activations) < 0.01 else 'NO'}")
    
    return {
        'losses': losses,
        'activations': activations,
        'grad_norms': grad_norms
    }

if __name__ == "__main__":
    compare_models() 