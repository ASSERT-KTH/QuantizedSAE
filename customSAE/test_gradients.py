import torch
import torch.nn as nn
import torch.nn.functional as F
from SAEs.binary_SAE import BinarySAE, binary_decoder
from nnba.adder import carry_save_adder
from nnba.logic import AND, OR, XOR

def test_logic_gates():
    print("\n=== Testing Logic Gates ===")
    # Test AND gate
    and_gate = AND()
    a = torch.tensor([0.7, 0.3, 0.8, 0.2], requires_grad=True)
    b = torch.tensor([0.6, 0.4, 0.2, 0.9], requires_grad=True)
    out = and_gate(a, b)
    loss = out.sum()
    loss.backward()
    print("AND Gate Gradients:")
    print(f"a.grad: {a.grad}")
    print(f"b.grad: {b.grad}")

def test_adder():
    print("\n=== Testing Carry Save Adder ===")
    csa = carry_save_adder(n_bits=4)
    
    # Test with just 2 numbers first (simpler case)
    x = torch.tensor([
        [[1, 0, 1, 0],  # First number
         [0, 1, 1, 0]]  # Second number
    ], dtype=torch.float, requires_grad=True)
    
    print("Input shape:", x.shape)
    print("Input requires_grad:", x.requires_grad)
    
    sum_out, carry = csa(x)
    print("Sum output shape:", sum_out.shape)
    print("Sum output requires_grad:", sum_out.requires_grad)
    print("Sum output:", sum_out)
    
    # Compute MSE loss properly
    target = torch.zeros_like(sum_out)
    loss = F.mse_loss(sum_out, target)
    print("Loss:", loss)
    print("Loss requires_grad:", loss.requires_grad)
    
    loss.backward()
    print("CSA Gradients:")
    print(f"Input grad norm: {x.grad.norm()}")
    print(f"Input grad: {x.grad}")
    
    # Now test with 3 numbers
    print("\nTesting with 3 numbers:")
    x = torch.tensor([
        [[1, 0, 1, 0],  # First number
         [0, 1, 1, 0],  # Second number
         [1, 1, 0, 1]]  # Third number
    ], dtype=torch.float, requires_grad=True)
    
    sum_out, carry = csa(x)
    print("Sum output:", sum_out)
    
    target = torch.zeros_like(sum_out)
    loss = F.mse_loss(sum_out, target)
    print("Loss:", loss)
    
    loss.backward()
    print("CSA Gradients:")
    print(f"Input grad norm: {x.grad.norm()}")
    print(f"Input grad: {x.grad}")

def test_binary_decoder():
    print("\n=== Testing Binary Decoder ===")
    decoder = binary_decoder(in_features=4, out_features=3, n_bits=4)
    x = torch.tensor([0.7, 0.3, 0.8, 0.2], requires_grad=True)
    sum_out, carry = decoder(x)
    loss = sum_out.sum() + carry.sum()
    loss.backward()
    print("Decoder Gradients:")
    print(f"Input grad norm: {x.grad.norm()}")
    print(f"Weight grad norm: {decoder.weight.grad.norm()}")
    print(f"Sum output shape: {sum_out.shape}")
    print(f"Carry output shape: {carry.shape}")

def test_full_network():
    print("\n=== Testing Full BinarySAE Network ===")
    model = BinarySAE(input_dim=4, hidden_dim=3, n_bits=4)
    x = torch.tensor([
        [1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1],  # 4 numbers, 4 bits each
        [0, 1, 1, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 0]
    ], dtype=torch.float, requires_grad=True)
    
    latent, recon, carry = model(x)
    loss = recon.sum() + carry.sum()
    loss.backward()
    
    print("Full Network Gradients:")
    print(f"Input grad norm: {x.grad.norm()}")
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            print(f"{name} grad norm: {param.grad.norm()}")
    
    print("\nOutput shapes:")
    print(f"Latent shape: {latent.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Carry shape: {carry.shape}")

if __name__ == "__main__":
    print("Starting gradient tests...")
    
    # Test each component
    test_logic_gates()
    test_adder()
    test_binary_decoder()
    test_full_network()
    
    print("\nAll tests completed!") 