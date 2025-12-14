"""
Sparse Autoencoder Implementations

This module contains various SAE architectures including baseline, binary,
ternary, and quantized variants.
"""

from .base import SparseAutoencoder
from .baseline import BaselineSparseAutoencoder
from .binary import BinarySAE
from .binary_latent import BinaryLatentSAE
from .ternary import TernarySparseAutoencoder
from .quantized_matryoshka import QuantizedMatryoshkaSAE
from .residual_quantized import ResidualQuantizedSAE

__all__ = [
    "SparseAutoencoder",
    "BaselineSparseAutoencoder",
    "BinarySAE",
    "BinaryLatentSAE",
    "TernarySparseAutoencoder",
    "QuantizedMatryoshkaSAE",
    "ResidualQuantizedSAE",
]