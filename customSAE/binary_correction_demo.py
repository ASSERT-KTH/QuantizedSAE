#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.getcwd())
from SAEs.binary_SAE import BinarySAE

def demonstrate_binary_correction_rule():
    """Demonstrate the binary correction rule with a concrete example"""
    print("🔧 Binary Correction Rule Demonstration")
    print("=" * 60)
    
    # Create a simple model
    model = BinarySAE(input_dim=2, hidden_dim=2, n_bits=4)
    
    # Set up a specific weight pattern to demonstrate the rule
    # Let's create an atom [0, 1, 0, 1] as in your example
    with torch.no_grad():
        # Row 0, atom 0 (columns 0-3): [0, 1, 0, 1]
        model.decoder.weight.data[0, 0] = 0.3  # bit 0: below threshold (0)
        model.decoder.weight.data[0, 1] = 0.7  # bit 1: above threshold (1) 
        model.decoder.weight.data[0, 2] = 0.4  # bit 2: below threshold (0)
        model.decoder.weight.data[0, 3] = 0.8  # bit 3: above threshold (1)
        
        # Row 0, atom 1 (columns 4-7): [1, 0, 1, 0] 
        model.decoder.weight.data[0, 4] = 0.6  # bit 0: above threshold (1)
        model.decoder.weight.data[0, 5] = 0.3  # bit 1: below threshold (0)
        model.decoder.weight.data[0, 6] = 0.7  # bit 2: above threshold (1)
        model.decoder.weight.data[0, 7] = 0.4  # bit 3: below threshold (0)
    
    def print_atom_state(row, atom_idx, title):
        """Helper function to print the binary state of an atom"""
        start_col = atom_idx * model.decoder.n_bits
        weights = model.decoder.weight.data[row, start_col:start_col+model.decoder.n_bits]
        binary = (weights >= model.decoder.threshold).int()
        print(f"{title}")
        print(f"  Weights: {weights.tolist()}")
        print(f"  Binary:  {binary.tolist()} (LSB to MSB: bit0, bit1, bit2, bit3)")
    
    print("📊 Initial State:")
    print_atom_state(0, 0, "Row 0, Atom 0:")
    print_atom_state(0, 1, "Row 0, Atom 1:")
    print()
    
    # Simulate a specific flip scenario from your example
    print("🎯 Simulating your example: [0,1,0,1] → [0,1,0,0]")
    print("   (bit 3 flips from 1→0, so MSB=3, correction_value=1)")
    
    # Store pre-update state
    model.store_decoder_pre_update_state()
    
    # Manually simulate the flip: bit 3 goes from 1→0
    with torch.no_grad():
        model.decoder.weight.data[0, 3] = 0.3  # Flip bit 3 from above to below threshold
    
    # Check for crossings
    crossing_indices, crossing_directions = model.check_decoder_threshold_crossings()
    
    print(f"🔍 Detected crossings: {crossing_indices}")
    print(f"🔍 Directions: {crossing_directions}")
    print()
    
    print("⚡ Applying Binary Correction Rule...")
    model.apply_decoder_binary_correction(crossing_indices, crossing_directions)
    print()
    
    print("📊 After Correction:")
    print_atom_state(0, 0, "Row 0, Atom 0:")
    print("   Expected: [1,1,1,0] (bits 0,1,2 set to 1, bit 3 stays 0)")
    print()
    
    # Test another scenario
    print("🎯 Testing another scenario: Flip bit 2 from 1→0 in atom 1")
    
    # Reset atom 1 and store state
    with torch.no_grad():
        model.decoder.weight.data[0, 4] = 0.6  # bit 0: 1
        model.decoder.weight.data[0, 5] = 0.3  # bit 1: 0  
        model.decoder.weight.data[0, 6] = 0.7  # bit 2: 1
        model.decoder.weight.data[0, 7] = 0.4  # bit 3: 0
    
    print_atom_state(0, 1, "Before: Row 0, Atom 1:")
    
    model.store_decoder_pre_update_state()
    
    # Flip bit 2 from 1→0
    with torch.no_grad():
        model.decoder.weight.data[0, 6] = 0.3  # bit 2: 1→0
    
    crossing_indices, crossing_directions = model.check_decoder_threshold_crossings()
    print(f"🔍 Detected crossings: {crossing_indices}")
    
    model.apply_decoder_binary_correction(crossing_indices, crossing_directions)
    print_atom_state(0, 1, "After: Row 0, Atom 1:")
    print("   Expected: [1,1,0,0] (bits 0,1 set to 1, bit 2 stays 0)")

def create_training_with_correction():
    """Show how to integrate binary correction into training"""
    print("\n" + "="*60)
    print("🚀 Training Integration Example")
    print("="*60)
    
    def binary_correction_callback(indices, directions, prev_values, current_values):
        """Callback that logs threshold crossings in terms of atoms and bits"""
        print(f"🚨 Threshold crossings detected at {len(indices)} positions!")
        for i, (idx, direction, prev_val, curr_val) in enumerate(zip(indices, directions, prev_values, current_values)):
            direction_str = "BELOW ➜ ABOVE" if direction else "ABOVE ➜ BELOW"
            row, col = idx[0].item(), idx[1].item()
            atom_idx = col // 4  # n_bits = 4
            bit_idx = col % 4
            print(f"  📍 Row {row}, Atom {atom_idx}, Bit {bit_idx}: {direction_str}")
            print(f"    📊 {prev_val:.4f} ➜ {curr_val:.4f}")
    
    model = BinarySAE(input_dim=2, hidden_dim=2, n_bits=4)
    model.set_decoder_threshold_crossing_callback(binary_correction_callback)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Generate dummy data
    torch.manual_seed(42)
    input_data = torch.randn(4, 8)  # 4 samples, 8 features
    true_sum = torch.zeros(4, 2)
    true_carry = torch.zeros(4, 2)
    target = torch.randn(4, 2)
    
    print("Training loop with automatic binary correction:")
    for epoch in range(3):
        print(f"\n📈 Epoch {epoch + 1}")
        
        # Store state before update
        model.store_decoder_pre_update_state()
        
        # Training step
        binary_latent, trigger = model(input_data, true_sum, true_carry)
        loss = nn.MSELoss()(trigger, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Check and apply corrections
        crossing_indices, crossing_directions = model.check_decoder_threshold_crossings()
        if crossing_indices is not None:
            print("   Applying binary correction rule...")
            model.apply_decoder_binary_correction(crossing_indices, crossing_directions)
        else:
            print("   ✅ No threshold crossings detected")

if __name__ == "__main__":
    demonstrate_binary_correction_rule()
    create_training_with_correction() 