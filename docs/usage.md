# Usage Guide

## Basic Usage

### Installing the Package

```bash
# Clone the repository
git clone https://github.com/yourusername/quantizedSAE.git
cd quantizedSAE

# Install in development mode
pip install -e .
```

### Training a SAE

```python
from quantized_sae.training.trainer import Trainer
from quantized_sae.data.dataset import HiddenStatesTorchDataset
from torch.utils.data import DataLoader

# Configuration
config = {
    "input_dim": 512,
    "hidden_dim": 32768,
    "lr": 1e-3,
    "batch_size": 4096,
    "n_bits": 4,  # For quantized variants
}

# Create trainer
trainer = Trainer(config, sae_type="binary_sae")

# Load data
dataset = HiddenStatesTorchDataset("path/to/hidden_states.pt")
dataloader = DataLoader(dataset, batch_size=config["batch_size"])

# Train
trainer.train(dataloader, epochs=100)
```

### Using Pre-trained SAEs

```python
from quantized_sae.inference.framework import load_sae

# Load a trained SAE
sae = load_sae("path/to/trained_sae.pth")

# Use for inference
latent, reconstruction = sae(input_tensor)
```

## Command Line Scripts

### Training Scripts

```bash
# Train binary SAE
cd scripts/training
python train_binary.py

# Train quantized SAE
python train_quantized.py
```

### Analysis Scripts

```bash
# Analyze SAE performance
cd scripts/analysis
python analyze_sae.py --model_path ../../models/trained_sae.pth

# Run dynamic analysis
python dynamic_analyze.py --model_path ../../models/trained_sae.pth
```

## Configuration Examples

### Binary SAE Configuration

```python
config = {
    "input_dim": 512,
    "hidden_dim": 32768,
    "lr": 1e-3,
    "batch_size": 4096,
    "n_bits": 1,
    "top_k": 32,
}
```

### Quantized Matryoshka SAE Configuration

```python
config = {
    "input_dim": 512,
    "hidden_dim": 32768,
    "lr": 1e-3,
    "batch_size": 4096,
    "n_bits": 4,
    "top_k": 32,
    "gamma": 0.1,
}
```

## Advanced Usage

### Custom SAE Implementation

```python
from quantized_sae.sae.base import SparseAutoencoder
import torch.nn as nn

class CustomSAE(SparseAutoencoder):
    def __init__(self, input_dim, hidden_dim):
        super().__init__(input_dim, hidden_dim)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU()
        )
        self.decoder = nn.Linear(hidden_dim, input_dim)
```

### Custom Training Loop

```python
from quantized_sae.sae.binary import BinarySAE
import torch.optim as optim

model = BinarySAE(input_dim=512, hidden_dim=32768, n_bits=4)
optimizer = optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    latent, reconstruction = model(batch)
    loss = compute_loss(reconstruction, batch)  # Your loss function
    loss.backward()
    optimizer.step()
```

## Best Practices

### Training Tips

1. **Learning Rate**: Start with 1e-3, reduce if training is unstable
2. **Batch Size**: Larger batches (4096+) generally work better
3. **Sparsity**: Adjust `top_k` based on your compression needs
4. **Quantization**: Higher `n_bits` gives better quality but less compression

### Evaluation Metrics

- **Reconstruction Error**: MSE between input and reconstruction
- **Sparsity**: Average number of active neurons per sample
- **Quantization Error**: Difference between full-precision and quantized weights
- **Gradient Flow**: Monitor gradient norms during training

### Debugging

- Use `torch.autograd.detect_anomaly()` for NaN/inf detection
- Monitor gradient norms with wandb/tensorboard
- Check sparsity patterns with analysis scripts