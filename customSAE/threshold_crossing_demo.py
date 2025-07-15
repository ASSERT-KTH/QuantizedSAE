#!/usr/bin/env python3

import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.getcwd())
from SAEs.binary_SAE import BinarySAE

def threshold_crossing_callback(indices, directions, prev_values, current_values):
    """Callback function that gets triggered when parameters cross the 0.5 threshold"""
    print(f"🚨 Threshold crossings detected at {len(indices)} positions!")
    for i, (idx, direction, prev_val, curr_val) in enumerate(zip(indices, directions, prev_values, current_values)):
        direction_str = "BELOW ➜ ABOVE" if direction else "ABOVE ➜ BELOW"
        print(f"  📍 Position [{idx[0].item()}, {idx[1].item()}]: {direction_str} threshold (0.5)")
        print(f"    📊 {prev_val:.6f} ➜ {curr_val:.6f} (change: {curr_val-prev_val:+.6f})")
    print("-" * 60)

def demo_threshold_crossing_detection():
    """Demonstrate threshold crossing detection during training"""
    print("🔬 Threshold Crossing Detection Demo")
    print("=" * 60)
    
    # Setup model and optimizer
    model = BinarySAE(input_dim=2, hidden_dim=3, n_bits=2)
    model.set_decoder_threshold_crossing_callback(threshold_crossing_callback)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  # Higher LR to trigger crossings
    
    # Generate some dummy data
    torch.manual_seed(42)
    input_data = torch.randn(4, 4)  # 4 samples, 4 features (2*n_bits)
    true_sum = torch.zeros(4, 2)  # dummy values
    true_carry = torch.zeros(4, 2)  # dummy values
    target = torch.randn(4, 2)  # dummy target for loss
    
    print(f"🎯 Initial decoder weights around threshold:")
    weights = model.decoder.weight.data
    near_threshold = torch.abs(weights - 0.5) < 0.1
    print(f"   Weights within 0.1 of threshold: {near_threshold.sum().item()}/{weights.numel()}")
    print(f"   Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print()
    
    # Training loop
    for epoch in range(10):
        print(f"📈 Epoch {epoch + 1}/10")
        
        # IMPORTANT: Store state before optimizer update
        model.store_decoder_pre_update_state()
        
        # Forward pass
        binary_latent, trigger = model(input_data, true_sum, true_carry)
        
        # Simple reconstruction loss (for demo purposes)
        loss = nn.MSELoss()(trigger, target)
        print(f"   Loss: {loss.item():.6f}")
        
        # Backward pass and optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # IMPORTANT: Check for threshold crossings after update
        crossing_indices, crossing_directions = model.check_decoder_threshold_crossings()
        
        if crossing_indices is None:
            print("   ✅ No threshold crossings detected")
        else:
            print(f"   ⚡ {len(crossing_indices)} parameters crossed threshold!")
        
        print()

def analyze_weight_distribution():
    """Analyze weight distribution relative to threshold"""
    print("📊 Weight Distribution Analysis")
    print("=" * 40)
    
    model = BinarySAE(input_dim=2, hidden_dim=3, n_bits=2)
    weights = model.decoder.weight.data
    
    below_threshold = (weights < 0.5).sum().item()
    above_threshold = (weights >= 0.5).sum().item()
    total = weights.numel()
    
    print(f"Total parameters: {total}")
    print(f"Below threshold (<0.5): {below_threshold} ({below_threshold/total*100:.1f}%)")
    print(f"Above threshold (≥0.5): {above_threshold} ({above_threshold/total*100:.1f}%)")
    print(f"Weight statistics: mean={weights.mean():.4f}, std={weights.std():.4f}")

if __name__ == "__main__":
    analyze_weight_distribution()
    print("\n")
    demo_threshold_crossing_detection() 