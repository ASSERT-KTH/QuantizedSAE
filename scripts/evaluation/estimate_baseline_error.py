import torch
import os
from hidden_state_dataset import HiddenStatesTorchDataset
from torch.utils.data import DataLoader
import numpy as np

def estimate_baseline_error(num_files=10):
    """
    Estimate the baseline error (variance) of the hidden states.
    This represents the MSE loss if there was no reconstruction (predicting zeros).
    """
    
    # Get chunk files
    chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt')]
    chunk_files = sorted(chunk_files)[:num_files]  # Use first num_files files
    
    print(f"Computing baseline error using {len(chunk_files)} files...")
    
    total_samples = 0
    total_squared_sum = 0.0
    total_sum = 0.0
    
    for i, f in enumerate(chunk_files):
        print(f"Processing file {i+1}/{len(chunk_files)}: {f}")
        
        dataset = HiddenStatesTorchDataset(os.path.join("dataset/", f))
        dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0)
        
        for batch in dataloader:
            batch = batch.float()
            
            # Skip NaN batches
            if torch.isnan(batch).any():
                continue
            
            # Compute sum and squared sum for variance calculation
            batch_sum = batch.sum().item()
            batch_squared_sum = (batch ** 2).sum().item()
            batch_samples = batch.numel()
            
            total_sum += batch_sum
            total_squared_sum += batch_squared_sum
            total_samples += batch_samples
    
    # Calculate mean and variance
    mean = total_sum / total_samples
    variance = (total_squared_sum / total_samples) - (mean ** 2)
    
    # MSE when predicting zeros (baseline error)
    baseline_mse_zeros = total_squared_sum / total_samples
    
    # MSE when predicting mean (minimum possible MSE for constant prediction)
    baseline_mse_mean = variance
    
    print("\n" + "="*60)
    print("BASELINE ERROR ESTIMATION RESULTS")
    print("="*60)
    print(f"Total samples analyzed: {total_samples:,}")
    print(f"Mean of hidden states: {mean:.6f}")
    print(f"Variance of hidden states: {variance:.6f}")
    print(f"Standard deviation: {np.sqrt(variance):.6f}")
    print(f"\nBaseline MSE (predicting zeros): {baseline_mse_zeros:.6f}")
    print(f"Baseline MSE (predicting mean): {baseline_mse_mean:.6f}")
    print("="*60)
    print(f"\nInterpretation:")
    print(f"- If your SAE achieves MSE < {baseline_mse_zeros:.6f}, it's doing better than no reconstruction")
    print(f"- If your SAE achieves MSE < {baseline_mse_mean:.6f}, it's doing better than predicting the mean")
    print(f"- The lower the MSE compared to these baselines, the better the reconstruction quality")
    
    return {
        'mean': mean,
        'variance': variance,
        'baseline_mse_zeros': baseline_mse_zeros,
        'baseline_mse_mean': baseline_mse_mean,
        'total_samples': total_samples
    }

if __name__ == "__main__":
    # Estimate using first 10 files (can be adjusted)
    results = estimate_baseline_error(num_files=10)

