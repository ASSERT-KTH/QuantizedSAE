# Architecture Overview

## Package Structure

The `quantized_sae` package is organized into several modules:

### Core Modules (`src/quantized_sae/`)

- **`sae/`**: Contains all SAE implementations
  - `base.py`: Abstract base class for SAEs
  - `baseline.py`: Standard sparse autoencoder
  - `binary.py`: Binary weight quantization
  - `ternary.py`: Ternary weight quantization with sparsity
  - `quantized_matryoshka.py`: Hierarchical quantization
  - `residual_quantized.py`: Residual quantization approach

- **`training/`**: Training utilities and trainer classes
  - `trainer.py`: Main training loop and utilities

- **`data/`**: Data loading and preprocessing
  - `dataset.py`: PyTorch dataset for hidden states
  - `loader.py`: Data loading utilities
  - `load_baseline.py`: Baseline model loading
  - `load_model.py`: General model loading from HuggingFace

- **`inference/`**: Inference and model wrapping utilities
  - `framework.py`: SAE wrapper for inference

- **`utils/`**: General utilities
  - `inspector.py`: Model inspection tools
  - `transformer_inspector.py`: Transformer-specific inspection
  - `encoder_debug.py`: Debugging utilities

## SAE Architecture Details

### Common Interface

All SAE implementations follow a similar structure:

```python
class SomeSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        # Initialize encoder and decoder

    def forward(self, x):
        # Encode input to sparse representation
        latent = self.encode(x)
        # Decode back to reconstruction
        reconstruction = self.decode(latent)
        return latent, reconstruction
```

### Baseline SAE

The baseline implementation uses:
- Linear encoder with ReLU activation
- Linear decoder
- Standard sparse autoencoder objective

### Quantized Variants

The quantized SAEs implement different quantization strategies:

- **Binary SAE**: Weights quantized to {-1, +1}
- **Ternary SAE**: Weights quantized to {-1, 0, +1} with learned sparsity masks
- **Quantized Matryoshka**: Hierarchical quantization allowing different precisions
- **Residual Quantized**: Residual connections between quantization levels

## Training Pipeline

The training pipeline consists of:

1. **Data Loading**: Load pre-computed hidden states from transformer models
2. **Model Initialization**: Create SAE instance with appropriate configuration
3. **Training Loop**: Alternating between reconstruction and sparsity objectives
4. **Evaluation**: Monitor reconstruction quality and sparsity metrics
5. **Model Saving**: Save trained weights for later use

## Key Design Decisions

### Modular Architecture
- Clear separation between different SAE variants
- Reusable training and data loading components
- Easy extension with new quantization methods

### Configuration-Driven
- All hyperparameters passed via configuration dictionaries
- Easy experimentation with different settings
- Reproducible experiments

### PyTorch Native
- Built on PyTorch for maximum flexibility
- Support for GPU acceleration
- Integration with existing ML pipelines