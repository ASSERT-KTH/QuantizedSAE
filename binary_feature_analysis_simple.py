#!/usr/bin/env python3
"""
Binary Feature Decomposition Analysis for SAE Feature Absorption
This explores the idea that binary representation provides insight into feature absorption.
"""

def integer_to_binary(values, bits_per_dim=4):
    """Convert integer values to binary representation."""
    binary = []
    for val in values:
        # Handle negative numbers with two's complement
        if val < 0:
            val = (1 << bits_per_dim) + val
        binary.extend([int(b) for b in format(val, f'0{bits_per_dim}b')])
    return binary

def binary_to_integer(binary_list, bits_per_dim=4):
    """Convert binary representation back to integers."""
    n_dims = len(binary_list) // bits_per_dim
    integers = []
    for i in range(n_dims):
        bits = binary_list[i*bits_per_dim:(i+1)*bits_per_dim]
        val = int(''.join(map(str, bits)), 2)
        # Handle two's complement for negative numbers
        if bits[0] == 1 and bits_per_dim > 1:
            val = val - (1 << bits_per_dim)
        integers.append(val)
    return integers

def binary_to_sparse_features(binary_vector):
    """Convert a binary vector to its extreme sparse decomposition."""
    features = []
    for i, bit in enumerate(binary_vector):
        if bit == 1:
            feature = [0] * len(binary_vector)
            feature[i] = 1
            features.append(feature)
    return features

def analyze_decomposition_complexity(binary_vector):
    """Analyze the complexity of different decompositions."""
    n_ones = sum(binary_vector)
    n_zeros = len(binary_vector) - n_ones
    
    analysis = {
        'binary_length': len(binary_vector),
        'active_bits': n_ones,
        'inactive_bits': n_zeros,
        'sparsity': n_ones / len(binary_vector) if len(binary_vector) > 0 else 0,
        'max_subfeatures': n_ones,  # In extreme sparse case
        'min_subfeatures': 1 if n_ones > 0 else 0  # All combined
    }
    
    return analysis

def compare_radix_systems(max_value=256):
    """Compare different radix systems for representation efficiency."""
    import math
    
    radices = [2, 3, 4, 8, 10, 16]
    comparison = []
    
    for radix in radices:
        digits_needed = math.ceil(math.log(max_value) / math.log(radix))
        # Maximum sparse features = digits * (radix - 1)
        # This is because each digit can be decomposed into at most (radix-1) features
        max_sparse_features = digits_needed * (radix - 1)
        efficiency = max_value / max_sparse_features if max_sparse_features > 0 else 0
        
        comparison.append({
            'radix': radix,
            'digits_needed': digits_needed,
            'max_sparse_features': max_sparse_features,
            'efficiency': efficiency,
            'bits_per_digit': math.log2(radix)
        })
    
    return comparison

def demonstrate_feature_absorption():
    """Show how binary decomposition relates to feature absorption."""
    # Example: Two semantic features that might be absorbed
    print("\n=== Feature Absorption Example ===")
    
    # Feature 1: "cat-like" (positions 1,2 active)
    feature1 = [0, 1, 1, 0, 0, 0, 0, 0]
    # Feature 2: "furry" (positions 4,5 active)
    feature2 = [0, 0, 0, 0, 1, 1, 0, 0]
    
    # Absorbed feature (when SAE combines them)
    absorbed = [feature1[i] or feature2[i] for i in range(len(feature1))]
    
    print(f"Feature 1 (e.g., 'cat-like'): {feature1}")
    print(f"Feature 2 (e.g., 'furry'):    {feature2}")
    print(f"Absorbed feature:             {absorbed}")
    
    # Show decomposition
    sparse_decomp = binary_to_sparse_features(absorbed)
    print(f"\nExtreme sparse decomposition ({len(sparse_decomp)} sub-features):")
    for i, feat in enumerate(sparse_decomp):
        print(f"  Sub-feature {i+1}: {feat}")
    
    return absorbed, sparse_decomp

def theoretical_insights():
    """Explain theoretical reasons why binary might be fundamental."""
    insights = """
=== Theoretical Insights: Why Binary Representation Matters ===

1. **Information Theory**: 
   - Binary provides optimal bit-per-choice encoding
   - Each bit carries exactly 1 bit of information
   - No redundancy in the representation

2. **Decomposability**:
   - Binary features are maximally decomposable
   - Each active bit represents an independent, atomic feature
   - No intermediate states between 0 and 1

3. **Orthogonality**:
   - Binary basis vectors are perfectly orthogonal
   - No overlap or interference between features
   - Clean separation of sub-features

4. **Sparsity Control**:
   - The number of 1s directly controls sparsity
   - Feature absorption happens when we can't afford to represent all 1s separately
   - Binary makes this trade-off explicit

5. **Biological Plausibility**:
   - Neurons exhibit binary-like behavior (firing/not firing)
   - Feature detection in neural networks often becomes binary-like
   - Matches the discrete nature of many cognitive features

6. **Computational Efficiency**:
   - Binary operations are the fastest in digital systems
   - Minimal memory footprint
   - Hardware optimization for binary computation
"""
    return insights

# Main execution
if __name__ == "__main__":
    print("=== Binary Feature Decomposition Analysis for SAE ===\n")
    
    # Example from the user's query
    integer_values = [6, -8]
    binary_vector = integer_to_binary(integer_values)
    
    print(f"Integer representation: {integer_values}")
    print(f"Binary representation: {binary_vector}")
    print(f"Binary as string: {''.join(map(str, binary_vector))}")
    
    # Verify conversion
    back_to_int = binary_to_integer(binary_vector)
    print(f"Converted back to integer: {back_to_int}")
    
    # Analyze the binary vector
    analysis = analyze_decomposition_complexity(binary_vector)
    print(f"\nBinary Vector Analysis:")
    for key, value in analysis.items():
        print(f"  {key}: {value}")
    
    # Show extreme sparse decomposition
    sparse_features = binary_to_sparse_features(binary_vector)
    print(f"\nExtreme sparse decomposition ({len(sparse_features)} features):")
    for i, feature in enumerate(sparse_features):
        print(f"  Feature {i+1}: {feature}")
    
    # Compare radix systems
    print("\n=== Radix System Comparison ===")
    radix_comparison = compare_radix_systems()
    print(f"{'Radix':<8} {'Digits':<8} {'Max Features':<15} {'Efficiency':<12} {'Bits/Digit':<12}")
    print("-" * 65)
    for system in radix_comparison:
        print(f"{system['radix']:<8} {system['digits_needed']:<8} "
              f"{system['max_sparse_features']:<15} {system['efficiency']:<12.2f} "
              f"{system['bits_per_digit']:<12.2f}")
    
    # Demonstrate feature absorption
    absorbed, decomp = demonstrate_feature_absorption()
    
    # Print theoretical insights
    print(theoretical_insights())
    
    # Additional analysis specific to SAE
    print("\n=== Implications for Sparse Autoencoders (SAE) ===")
    print("""
1. **Feature Absorption as Binary Overlap**:
   - When sparsity penalty is high, SAE combines features with overlapping binary patterns
   - The absorbed feature represents the OR operation of sub-features
   - Binary decomposition reveals the minimum number of truly independent features

2. **Sparsity vs. Decomposition Trade-off**:
   - Each active bit in binary representation is a potential independent feature
   - SAE must balance between representing each bit separately vs. combining them
   - The sparsity parameter directly controls this trade-off

3. **Optimal Dictionary Size**:
   - For n-bit binary patterns, we need at most n features for complete decomposition
   - In practice, features have varying numbers of active bits
   - Dictionary size should scale with the average binary complexity of features

4. **Why Not Other Radices?**:
   - Higher radices (3, 4, etc.) introduce ambiguity in decomposition
   - Example: In base-3, does '2' decompose to '1+1' or stay as '2'?
   - Binary eliminates this ambiguity: 1 can only decompose to 1

5. **Practical Implications**:
   - Monitor the binary sparsity of learned features
   - Features with many active bits are candidates for decomposition
   - Adjust sparsity penalty based on desired decomposition level
""")
    
    # Show how different representations of the same value relate to absorption
    print("\n=== Alternative Decompositions and Absorption ===")
    print("For [6, -8] = [0110, 1000] in binary:")
    print("1. Extreme sparse: 3 one-hot vectors (positions 1, 2, 4)")
    print("2. Partially absorbed: Could combine positions 1&2 into one feature")
    print("3. Fully absorbed: Single feature representing the entire pattern")
    print("\nThe key insight: Binary representation reveals the natural decomposition hierarchy!")