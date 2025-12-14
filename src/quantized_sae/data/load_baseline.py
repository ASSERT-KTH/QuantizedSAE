import torch
import struct
import json

def load_safetensors(filepath):
    """
    Load a safetensors file and return a dictionary of tensors.
    
    Args:
        filepath: Path to the .safetensors file
    
    Returns:
        dict: Dictionary mapping tensor names to torch tensors
    """
    tensors = {}
    
    with open(filepath, 'rb') as f:
        # Read header length (first 8 bytes, little-endian uint64)
        header_size = struct.unpack('<Q', f.read(8))[0]
        
        # Read and parse JSON header
        header = json.loads(f.read(header_size).decode('utf-8'))
        
        # Map safetensors dtype to torch dtype
        dtype_map = {
            'F32': torch.float32,
            'F16': torch.float16,
            'BF16': torch.bfloat16,
            'I32': torch.int32,
            'I64': torch.int64,
        }
        
        # Load each tensor
        for key, info in header.items():
            if key == '__metadata__':
                continue
            
            offset_start, offset_end = info['data_offsets']
            dtype = dtype_map.get(info['dtype'], torch.float32)
            shape = info['shape']
            
            # Read tensor data
            f.seek(8 + header_size + offset_start)
            tensor_bytes = f.read(offset_end - offset_start)
            
            # Convert to torch tensor
            tensor = torch.frombuffer(tensor_bytes, dtype=dtype).clone()
            if shape:
                tensor = tensor.reshape(shape)
            
            tensors[key] = tensor
    
    return tensors

def compute_cosine_similarity_matrix(W, normalize=True):
    """
    Efficiently compute pairwise cosine similarities between rows of W.
    
    Args:
        W: Tensor of shape [n_features, d_model]
        normalize: If True, assumes W is already normalized. Otherwise normalizes it.
    
    Returns:
        Cosine similarity matrix of shape [n_features, n_features]
    """
    if normalize:
        # Normalize rows to unit vectors
        W_norm = W / W.norm(dim=1, keepdim=True)
    else:
        W_norm = W
    
    # Cosine similarity = normalized dot product
    # This is much faster than nested loops!
    cosine_sim = W_norm @ W_norm.T
    
    return cosine_sim


def analyze_cosine_similarities(W, sample_size=None):
    """
    Analyze and visualize cosine similarity distribution.
    
    Args:
        W: Weight matrix [n_features, d_model]
        sample_size: If provided, only compute for a random subset of features
    """
    import numpy as np
    
    print(f"Computing cosine similarities for {W.shape[0]} features...")
    
    # For very large matrices, optionally sample
    if sample_size and W.shape[0] > sample_size:
        print(f"Sampling {sample_size} features for efficiency...")
        indices = torch.randperm(W.shape[0])[:sample_size]
        W_sample = W[indices]
    else:
        W_sample = W
    
    # Compute cosine similarity matrix
    cosine_sim = compute_cosine_similarity_matrix(W_sample)
    
    # Get upper triangle (excluding diagonal) for statistics
    # We exclude diagonal because self-similarity is always 1
    mask = torch.triu(torch.ones_like(cosine_sim), diagonal=1).bool()
    similarities = cosine_sim[mask].cpu().numpy()
    
    print(f"\nCosine Similarity Statistics (excluding self-similarities):")
    print(f"  Mean: {similarities.mean():.4f}")
    print(f"  Std: {similarities.std():.4f}")
    print(f"  Min: {similarities.min():.4f}")
    print(f"  Max: {similarities.max():.4f}")
    print(f"  Median: {np.median(similarities):.4f}")
    print(f"  25th percentile: {np.percentile(similarities, 25):.4f}")
    print(f"  75th percentile: {np.percentile(similarities, 75):.4f}")
    
    # Count high similarities (potential redundancy)
    high_sim_threshold = 0.9
    high_sim_count = (similarities > high_sim_threshold).sum()
    total_pairs = len(similarities)
    print(f"\n  Pairs with similarity > {high_sim_threshold}: {high_sim_count} ({100*high_sim_count/total_pairs:.2f}%)")
    
    return cosine_sim, similarities


def plot_cosine_similarity_distribution(similarities, save_path=None):
    """
    Create visualizations of cosine similarity distribution.
    
    Args:
        similarities: 1D array of similarity values
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Histogram
    ax = axes[0, 0]
    ax.hist(similarities, bins=100, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Distribution of Pairwise Cosine Similarities(Baseline SAE)', fontsize=14, fontweight='bold')
    ax.axvline(similarities.mean(), color='red', linestyle='--', label=f'Mean: {similarities.mean():.3f}')
    ax.axvline(np.median(similarities), color='green', linestyle='--', label=f'Median: {np.median(similarities):.3f}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Log-scale histogram
    ax = axes[0, 1]
    ax.hist(similarities, bins=100, edgecolor='black', alpha=0.7, log=True)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Frequency (log scale)', fontsize=12)
    ax.set_title('Distribution (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 3. CDF (Cumulative Distribution)
    ax = axes[1, 0]
    sorted_sim = np.sort(similarities)
    cdf = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
    ax.plot(sorted_sim, cdf, linewidth=2)
    ax.set_xlabel('Cosine Similarity', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.axvline(np.median(similarities), color='red', linestyle='--', alpha=0.5, 
               label=f'Median: {np.median(similarities):.3f}')
    ax.legend()
    
    # 4. Box plot
    ax = axes[1, 1]
    bp = ax.boxplot([similarities], vert=True, patch_artist=True, widths=0.6)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Cosine Similarity', fontsize=12)
    ax.set_title('Box Plot of Similarities', fontsize=14, fontweight='bold')
    ax.set_xticklabels(['All Pairs'])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add statistics as text
    stats_text = f"n = {len(similarities):,}\nμ = {similarities.mean():.4f}\nσ = {similarities.std():.4f}"
    ax.text(1.15, 0.5, stats_text, transform=ax.transAxes, 
            verticalalignment='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFigure saved to: {save_path}")
    
    plt.savefig('cosine_similarity_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Figure saved to: cosine_similarity_distribution.png")
    plt.close()
    
    return fig


def plot_similarity_heatmap(cosine_sim, n_show=100, save_path=None):
    """
    Plot a heatmap of cosine similarities (for a subset of features).
    
    Args:
        cosine_sim: Full cosine similarity matrix
        n_show: Number of features to show in heatmap
        save_path: Optional path to save the figure
    """
    import matplotlib.pyplot as plt
    
    # Take a subset for visualization
    n = min(n_show, cosine_sim.shape[0])
    subset = cosine_sim[:n, :n].cpu().numpy()
    
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(subset, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    ax.set_xlabel(f'Feature Index (first {n})', fontsize=12)
    ax.set_ylabel(f'Feature Index (first {n})', fontsize=12)
    ax.set_title(f'Cosine Similarity Heatmap\n(First {n} Features)', 
                 fontsize=14, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Cosine Similarity', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    else:
        plt.savefig('cosine_similarity_heatmap.png', dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: cosine_similarity_heatmap.png")
    
    plt.close()
    
    return fig

# Example usage:
if __name__ == "__main__":
    # Load the baseline SAE
    sae_path = "./baseline_SAE/EleutherAI/sae-pythia-70m-32k/layers.3/sae.safetensors"
    sae_tensors = load_safetensors(sae_path)
    
    # Print available tensors
    print("Loaded SAE tensors:")
    for name, tensor in sae_tensors.items():
        print(f"  {name}: shape {tensor.shape}, dtype {tensor.dtype}")
    
    print("\n" + "="*60)
    
    # Access specific tensors
    W_dec = sae_tensors['W_dec']  # Decoder weight [32768, 512]
    b_dec = sae_tensors['b_dec']  # Decoder bias [512]
    W_enc = sae_tensors['encoder.weight']  # Encoder weight [32768, 512]
    b_enc = sae_tensors['encoder.bias']  # Encoder bias [32768]
    
    print("\n" + "="*60)
    print("\nAnalyzing cosine similarities between decoder features...")
    print("Note: Computing for all 32768 features. This may take a moment...")
    
    # Analyze similarities (use all features since decoder weights are normalized)
    cosine_sim_matrix, similarities = analyze_cosine_similarities(W_dec, sample_size=None)
    
    # Plot the distribution
    print("\nGenerating visualizations...")
    plot_cosine_similarity_distribution(similarities)
    
    # Plot a heatmap for a subset of features
    plot_similarity_heatmap(cosine_sim_matrix, n_show=200)
    
    print("\n" + "="*60)
    print("\nVisualization complete! Check the generated PNG files:")
    print("  - cosine_similarity_distribution.png (4-panel statistical plots)")
    print("  - cosine_similarity_heatmap.png (heatmap of first 200 features)")