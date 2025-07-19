import torch
import torch.nn as nn
from SAEs.binary_SAE import BinarySAE
import matplotlib.pyplot as plt

def test_simple_binary_sae():
    """Test if binary SAE can learn a simple identity mapping."""
    
    device = 'cpu'  # Force CPU for testing
    
    # Simple configuration
    input_dim = 8  # Small dimension for testing
    hidden_dim = 32  # Reasonable expansion
    n_bits = 8
    batch_size = 64
    
    # Create model
    model = BinarySAE(input_dim, hidden_dim, n_bits).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Generate simple binary data (8-bit signed integers)
    # Let's create some patterns
    losses = []
    
    for epoch in range(1000):
        # Generate random signed integers in range [-64, 63] (to avoid overflow)
        integers = torch.randint(-64, 64, (batch_size, input_dim), device=device).float()
        
        # Convert to binary representation (matching dataset's quantize_signed)
        scale_factor = 2**(n_bits - 1) / 2.0  # gamma=2
        scaled = torch.clamp(integers * scale_factor, -128, 127).round().int()
        
        # Convert to binary
        bit_positions = torch.arange(0, n_bits, device=device)
        mask = (1 << n_bits) - 1
        scaled_expanded = (scaled & mask).unsqueeze(-1)
        binary_input = ((scaled_expanded >> bit_positions) & 1).float()
        binary_input = binary_input.view(batch_size, -1)
        
        # Forward pass
        latent, loss = model(binary_input)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Check gradients
        if epoch == 0:
            # Get gradients - with weight norm we need to access differently
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
            print(f"Encoder grad norm: {enc_grad_norm:.6f}")
            print(f"Decoder grad norm: {dec_grad_norm:.6f}")
        
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            activated = latent.sum(dim=-1).mean()
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}, Activated = {activated:.1f}")
    
    # Plot loss curve
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.yscale('log')
    
    # Test reconstruction quality
    with torch.no_grad():
        # Generate test data
        test_integers = torch.tensor([[-64], [-32], [0], [32], [63]], device=device).float()
        test_integers = test_integers.repeat(1, input_dim)
        
        # Convert to binary
        scaled = torch.clamp(test_integers * scale_factor, -128, 127).round().int()
        scaled_expanded = (scaled & mask).unsqueeze(-1)
        test_binary = ((scaled_expanded >> bit_positions) & 1).float()
        test_binary = test_binary.view(test_integers.shape[0], -1)
        
        # Get latents
        latent, _ = model(test_binary)
        
        # Manually compute reconstruction
        prob_weights = torch.sigmoid(model.decoder.weight)
        hard_weights = (prob_weights > 0.5).float()
        
        powers = 2 ** torch.arange(n_bits, device=device).float()
        powers[-1] *= -1
        
        int_weights = (hard_weights.view(model.decoder.in_features, -1, n_bits) * powers).sum(-1)
        pred_ints = (latent.unsqueeze(-1) * int_weights.unsqueeze(0)).sum(-2)
        
        # Original integers from binary
        orig_ints = (test_binary.view(test_binary.shape[0], -1, n_bits) * powers).sum(-1)
        
        print("\nReconstruction test:")
        print("Original | Predicted | Error")
        for i in range(test_integers.shape[0]):
            orig = orig_ints[i, 0].item()
            pred = pred_ints[i, 0].item()
            print(f"{orig:8.1f} | {pred:9.1f} | {abs(orig-pred):5.1f}")
    
    plt.subplot(1, 2, 2)
    plt.bar(['Original', 'Predicted'], [orig_ints.mean().item(), pred_ints.mean().item()])
    plt.ylabel('Mean value')
    plt.title('Mean reconstruction')
    
    plt.tight_layout()
    plt.savefig('binary_sae_simple_test.png')
    print("\nPlot saved to binary_sae_simple_test.png")
    
    return model, losses

if __name__ == "__main__":
    model, losses = test_simple_binary_sae()
    
    # Check if model learned anything
    if losses[-1] < losses[0] * 0.1:
        print("\n✓ Model successfully reduced loss!")
    else:
        print("\n✗ Model failed to learn effectively")
        print(f"  Initial loss: {losses[0]:.4f}")
        print(f"  Final loss: {losses[-1]:.4f}") 