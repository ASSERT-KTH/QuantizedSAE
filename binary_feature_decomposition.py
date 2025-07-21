import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
import seaborn as sns

def binary_to_features(binary_vector):
    """Convert a binary vector to its sparse decomposition."""
    features = []
    for i, bit in enumerate(binary_vector):
        if bit == 1:
            feature = np.zeros_like(binary_vector)
            feature[i] = 1
            features.append(feature)
    return features

def integer_to_binary(values, bits_per_dim=4):
    """Convert integer values to binary representation."""
    binary = []
    for val in values:
        # Handle negative numbers with two's complement
        if val < 0:
            val = (1 << bits_per_dim) + val
        binary.extend([int(b) for b in format(val, f'0{bits_per_dim}b')])
    return np.array(binary)

def binary_to_integer(binary_vector, bits_per_dim=4):
    """Convert binary representation back to integers."""
    n_dims = len(binary_vector) // bits_per_dim
    integers = []
    for i in range(n_dims):
        bits = binary_vector[i*bits_per_dim:(i+1)*bits_per_dim]
        val = int(''.join(map(str, bits)), 2)
        # Handle two's complement for negative numbers
        if bits[0] == 1 and bits_per_dim > 1:
            val = val - (1 << bits_per_dim)
        integers.append(val)
    return integers

def analyze_sparsity_levels(binary_vector):
    """Analyze different sparsity levels of decomposition."""
    n_ones = np.sum(binary_vector)
    decompositions = {}
    
    # Extreme sparse (one hot per active bit)
    decompositions['extreme_sparse'] = binary_to_features(binary_vector)
    
    # Intermediate sparsity levels
    if n_ones > 2:
        for k in range(2, min(n_ones, 4)):
            decompositions[f'{k}_sparse'] = []
            active_indices = np.where(binary_vector == 1)[0]
            for combo in combinations(active_indices, k):
                feature = np.zeros_like(binary_vector)
                for idx in combo:
                    feature[idx] = 1
                decompositions[f'{k}_sparse'].append(feature)
    
    return decompositions

def visualize_decomposition(original_vector, decompositions):
    """Visualize the original vector and its decompositions."""
    fig, axes = plt.subplots(len(decompositions) + 1, 1, figsize=(12, 2 * (len(decompositions) + 1)))
    
    # Original vector
    ax = axes[0] if len(decompositions) > 0 else axes
    im = ax.imshow([original_vector], cmap='Blues', aspect='auto', vmin=0, vmax=1)
    ax.set_title('Original Binary Vector', fontsize=14)
    ax.set_yticks([])
    ax.set_xticks(range(len(original_vector)))
    
    # Add text annotations
    for i, val in enumerate(original_vector):
        ax.text(i, 0, str(int(val)), ha='center', va='center', 
                color='white' if val == 1 else 'black')
    
    # Decompositions
    for idx, (name, features) in enumerate(decompositions.items()):
        ax = axes[idx + 1]
        
        # Create matrix for visualization
        if features:
            feature_matrix = np.array(features)
            im = ax.imshow(feature_matrix, cmap='Blues', aspect='auto', vmin=0, vmax=1)
            ax.set_title(f'{name.replace("_", " ").title()} Decomposition ({len(features)} features)', 
                        fontsize=12)
            ax.set_ylabel('Features')
            ax.set_yticks(range(len(features)))
        else:
            ax.text(0.5, 0.5, 'No decomposition available', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'{name.replace("_", " ").title()} Decomposition', fontsize=12)
        
        ax.set_xticks(range(len(original_vector)))
    
    plt.tight_layout()
    return fig

def analyze_radix_efficiency():
    """Analyze why binary (radix-2) might be particularly efficient."""
    radices = [2, 3, 4, 8, 16]
    n_values = 256  # Total possible values
    
    results = {
        'radix': [],
        'bits_needed': [],
        'max_sparse_features': [],
        'efficiency': []
    }
    
    for radix in radices:
        bits_needed = int(np.ceil(np.log(n_values) / np.log(radix)))
        max_sparse_features = bits_needed * (radix - 1)
        efficiency = n_values / max_sparse_features
        
        results['radix'].append(radix)
        results['bits_needed'].append(bits_needed)
        results['max_sparse_features'].append(max_sparse_features)
        results['efficiency'].append(efficiency)
    
    return results

def plot_radix_analysis(results):
    """Plot the radix efficiency analysis."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Bits needed and max sparse features
    ax1.plot(results['radix'], results['bits_needed'], 'o-', label='Digits needed', markersize=8)
    ax1.plot(results['radix'], results['max_sparse_features'], 's-', 
             label='Max sparse features', markersize=8)
    ax1.set_xlabel('Radix')
    ax1.set_ylabel('Count')
    ax1.set_title('Representation Requirements by Radix')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(results['radix'])
    
    # Plot 2: Efficiency
    ax2.plot(results['radix'], results['efficiency'], 'o-', color='green', markersize=8)
    ax2.set_xlabel('Radix')
    ax2.set_ylabel('Efficiency (values per sparse feature)')
    ax2.set_title('Representation Efficiency by Radix')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(results['radix'])
    
    # Highlight binary
    ax2.axvline(x=2, color='red', linestyle='--', alpha=0.5, label='Binary (radix-2)')
    ax2.legend()
    
    plt.tight_layout()
    return fig

def demonstrate_feature_absorption():
    """Demonstrate how feature absorption relates to binary decomposition."""
    # Example: Two features that might be absorbed into one
    feature1 = np.array([0, 1, 1, 0, 0, 0, 0, 0])  # e.g., "cat-like"
    feature2 = np.array([0, 0, 0, 0, 1, 1, 0, 0])  # e.g., "furry"
    absorbed = feature1 + feature2  # Combined feature: "cat-like and furry"
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    
    # Visualize individual features
    for i, (feat, name) in enumerate([(feature1, 'Feature 1'), (feature2, 'Feature 2')]):
        ax = axes[0, i]
        im = ax.imshow([feat], cmap='Blues', aspect='auto', vmin=0, vmax=1)
        ax.set_title(name)
        ax.set_yticks([])
        for j, val in enumerate(feat):
            ax.text(j, 0, str(int(val)), ha='center', va='center',
                   color='white' if val == 1 else 'black')
    
    # Absorbed feature
    ax = axes[1, 0]
    im = ax.imshow([absorbed], cmap='Reds', aspect='auto', vmin=0, vmax=2)
    ax.set_title('Absorbed Feature (Combined)')
    ax.set_yticks([])
    for j, val in enumerate(absorbed):
        ax.text(j, 0, str(int(val)), ha='center', va='center',
               color='white' if val > 0 else 'black')
    
    # Binary decomposition of absorbed
    ax = axes[1, 1]
    binary_absorbed = (absorbed > 0).astype(int)
    decomp = binary_to_features(binary_absorbed)
    if decomp:
        decomp_matrix = np.array(decomp)
        im = ax.imshow(decomp_matrix, cmap='Greens', aspect='auto', vmin=0, vmax=1)
        ax.set_title(f'Binary Decomposition ({len(decomp)} sub-features)')
        ax.set_ylabel('Sub-features')
        ax.set_yticks(range(len(decomp)))
    
    plt.tight_layout()
    return fig

# Main execution
if __name__ == "__main__":
    # Example from the user's query
    print("=== Binary Feature Decomposition Analysis ===\n")
    
    # Convert [6, -8] to binary
    integer_values = [6, -8]
    binary_vector = integer_to_binary(integer_values)
    
    print(f"Integer representation: {integer_values}")
    print(f"Binary representation: {binary_vector}")
    print(f"Binary as string: {''.join(map(str, binary_vector))}")
    
    # Verify conversion
    back_to_int = binary_to_integer(binary_vector)
    print(f"Converted back to integer: {back_to_int}\n")
    
    # Analyze decompositions
    decompositions = analyze_sparsity_levels(binary_vector)
    
    print(f"Number of 1s in binary vector: {np.sum(binary_vector)}")
    print(f"Extreme sparse decomposition uses {len(decompositions['extreme_sparse'])} features")
    
    # Visualize decomposition
    fig1 = visualize_decomposition(binary_vector, {'extreme_sparse': decompositions['extreme_sparse']})
    plt.savefig('binary_decomposition.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Analyze radix efficiency
    print("\n=== Radix Efficiency Analysis ===")
    radix_results = analyze_radix_efficiency()
    
    for i in range(len(radix_results['radix'])):
        print(f"Radix-{radix_results['radix'][i]}: "
              f"{radix_results['bits_needed'][i]} digits needed, "
              f"max {radix_results['max_sparse_features'][i]} sparse features, "
              f"efficiency = {radix_results['efficiency'][i]:.2f}")
    
    fig2 = plot_radix_analysis(radix_results)
    plt.savefig('radix_efficiency.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Demonstrate feature absorption
    print("\n=== Feature Absorption Demonstration ===")
    fig3 = demonstrate_feature_absorption()
    plt.savefig('feature_absorption.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Additional analysis: Why binary might be fundamental
    print("\n=== Why Binary (Radix-2) Might Be Fundamental ===")
    print("1. Maximum decomposability: Each position is either on/off (no intermediate states)")
    print("2. Information theoretic optimality: Binary provides the most efficient bit-per-choice encoding")
    print("3. Orthogonality: Binary features are maximally orthogonal (no overlap in basis)")
    print("4. Biological plausibility: Neurons often exhibit binary-like firing patterns")
    print("5. Computational efficiency: Binary operations are fastest in digital systems")