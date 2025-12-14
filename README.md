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

### Training a Binary SAE

```bash
cd scripts/training
python train_binary.py
```

### Training a Quantized SAE

```bash
cd scripts/training
python train_quantized.py
```

### Running Analysis

```bash
cd scripts/analysis
python analyze_sae.py --model_path ../models/trained_sae.pth
```

## SAE Variants

### Baseline SAE
Standard sparse autoencoder implementation with ReLU activation and linear decoder.

### Binary SAE
Quantizes weights to binary values {-1, +1} for extreme compression.

### Ternary SAE
Quantizes weights to ternary values {-1, 0, +1} with learned sparsity masks.

### Quantized Matryoshka SAE
Hierarchical quantization that allows different precision levels for different bits.

### Residual Quantized SAE
Uses residual connections between quantization levels for better reconstruction.

## Configuration

Training configurations are specified as dictionaries with the following keys:
- `input_dim`: Input feature dimension
- `hidden_dim`: Hidden dimension (sparsity level)
- `lr`: Learning rate
- `batch_size`: Training batch size
- `n_bits`: Number of bits for quantization (for quantized variants)
- `top_k`: Sparsity level (for some variants)
- `gamma`: Quantization parameter

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
@misc{quantizedsae2024,
  title={Quantized Sparse Autoencoders},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/quantizedSAE}
}
```