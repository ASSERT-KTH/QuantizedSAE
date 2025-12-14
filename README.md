# Quantized Sparse Autoencoders (QSAE)

A collection of quantized sparse autoencoder implementations for efficient representation learning and compression.

## Overview

This repository contains implementations of various sparse autoencoder (SAE) architectures with quantization techniques to enable efficient inference and storage. The implementations include:

- **Baseline SAE**: Standard sparse autoencoder implementation
- **Binary SAE**: SAE with binary weight quantization
- **Ternary SAE**: SAE with ternary weight quantization
- **Quantized Matryoshka SAE**: Hierarchical quantization approach
- **Residual Quantized SAE**: Residual-based quantization method

## Repository Structure

```
quantizedSAE/
├── src/quantized_sae/          # Core package
│   ├── sae/                    # SAE implementations
│   ├── training/               # Training utilities
│   ├── data/                   # Data loading and processing
│   ├── inference/              # Inference utilities
│   └── utils/                  # General utilities
├── scripts/                    # Executable scripts
│   ├── training/               # Training scripts
│   ├── analysis/               # Analysis and evaluation scripts
│   └── evaluation/             # Evaluation scripts
├── experiments/                # Experiment configurations and results
├── models/                     # Pre-trained models and checkpoints
├── data/                       # Data storage (gitignored)
├── docs/                       # Documentation
├── tests/                      # Unit tests
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/quantizedSAE.git
cd quantizedSAE
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install the package:
```bash
pip install -e .
```

## Quick Start

### Training SAEs

The unified training script supports all SAE variants:

```bash
# Binary SAE with two's complement encoding
python scripts/training/train.py --sae_type b_sae --input_dim 512 --hidden_dim 32768 --n_bits 4 --data_path /path/to/data.pt

# Quantized Matryoshka SAE
python scripts/training/train.py --sae_type q_sae --input_dim 512 --hidden_dim 32768 --top_k 32 --n_bits 4 --data_path /path/to/data.pt

# Ternary SAE with learned sparsity
python scripts/training/train.py --sae_type t_sae --input_dim 512 --hidden_dim 32768 --data_path /path/to/data.pt

# Baseline SAE
python scripts/training/train.py --sae_type baseline_sae --input_dim 512 --hidden_dim 32768 --data_path /path/to/data.pt
```

### Running Analysis

```bash
cd scripts/analysis
python analyze_sae.py --model_path ../models/trained_sae.pth
```

## SAE Variants

### Baseline SAE
Standard sparse autoencoder implementation with:
- **Encoder**: Linear layer with ReLU activation
- **Decoder**: Linear layer
- **Sparsity**: Top-k activation selection
- **Use case**: Baseline for comparison, no quantization

### Binary SAE (Two's Complement Encoding)
Advanced binary quantization with integer representation:
- **Encoding**: Each weight represented by `n_bits` binary values (0/1)
- **Two's Complement**: Most significant bit (MSB) is negative, enabling signed integer representation
- **Integer Conversion**: Binary bits converted to integers using weighted sum: `Σ(bit_i × 2^i)` with `bit_{MSB} × (-2^{n_bits-1})`
- **Quantization Range**: Controlled by `gamma` parameter, mapping integers to floating-point values
- **Training**: Uses continuous sigmoid probabilities during training, thresholded to binary during inference
- **Use case**: Extreme compression while maintaining rich integer representation

**Example**: With 4 bits and gamma=4.0, binary pattern `[1,0,1,0]` becomes integer `1×(-8) + 0×4 + 1×2 + 0×1 = -6`, then scales to `-6 × (4.0/8) = -3.0`

### Ternary SAE (Learned Sparsity)
Ternary quantization with adaptive sparsity:
- **Weights**: Values constrained to {-1, 0, +1}
- **Sparsity**: Learned masks that dynamically prune weights during training
- **Dynamic Pruning**: Weights below threshold become zero, others quantized to ±1
- **Gradient Masks**: Prevents gradient flow through pruned connections
- **Use case**: Balances compression with sparsity for efficient inference

### Quantized Matryoshka SAE (Hierarchical Quantization)
Nested quantization with progressive precision:
- **Architecture**: Multiple quantization levels with increasing bit depths
- **Nested Dictionaries**: Each level refines the previous quantization
- **Hierarchical**: Lower bits provide coarse quantization, higher bits add precision
- **Top-k Integration**: Sparsity applied at each quantization level
- **Use case**: Variable precision - can use fewer bits for faster inference or more bits for accuracy

### Residual Quantized SAE
Residual connections across quantization levels:
- **Residual Learning**: Each quantization level corrects errors from previous levels
- **Accumulative Precision**: Errors compound to improve overall reconstruction
- **Bit-wise Residuals**: Each bit contributes to correcting the accumulated error
- **Use case**: Better reconstruction quality by leveraging error correction across quantization levels

## Configuration

Training configurations use the following parameters:

### Common Parameters
- `input_dim`: Input feature dimension (e.g., 512 for transformer hidden states)
- `hidden_dim`: Hidden dimension / dictionary size (e.g., 32768)
- `lr`: Learning rate (default: 1e-3)
- `batch_size`: Training batch size (default: 4096)
- `epochs`: Number of training epochs (default: 100)

### Quantized SAE Parameters
- `n_bits`: Number of bits for quantization (affects compression ratio and precision)
- `gamma`: Quantization range parameter (controls the floating-point range that integers map to)
  - For binary SAE: Maps quantized integers to `[-gamma, gamma]` range
  - For hierarchical SAEs: Controls absolute range of quantization levels
- `top_k`: Sparsity level for hierarchical SAEs (number of active features)

## Analysis Tools

The repository includes comprehensive analysis tools:

- **Gradient Analysis**: Monitor training stability and gradient flow
- **Quantization Error Analysis**: Evaluate reconstruction quality
- **Sparsity Analysis**: Analyze activation patterns
- **Cosine Similarity Analysis**: Study feature representations

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```
@misc{quantizedsae2025,
  title={Quantized Sparse Autoencoders},
  author={Tux},
  year={2025},
  url={https://github.com/yourusername/quantizedSAE}
}
```