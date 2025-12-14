#!/usr/bin/env python3
"""
Unified training script for all Quantized SAE architectures.

This script supports training various SAE variants including:
- baseline_sae: Standard sparse autoencoder
- b_sae: Binary SAE with two's complement encoding
- t_sae: Ternary SAE with {-1, 0, +1} weights
- bl_sae: Binary latent SAE
- q_sae: Quantized Matryoshka SAE (hierarchical quantization)
- rq_sae: Residual Quantized SAE

Usage:
    python train.py --sae_type b_sae --input_dim 512 --hidden_dim 32768 --n_bits 4
    python train.py --sae_type q_sae --input_dim 512 --hidden_dim 32768 --top_k 32 --n_bits 4
"""

import sys
import os
import argparse
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from quantized_sae.data.dataset import HiddenStatesTorchDataset
import wandb  # Will be imported conditionally if needed
import time
import math
from quantized_sae.sae.ternary import TernarySparseAutoencoder
from quantized_sae.sae.binary_latent import BinaryLatentSAE
from quantized_sae.sae.binary import BinarySAE
from quantized_sae.sae.quantized_matryoshka import QuantizedMatryoshkaSAE
from quantized_sae.sae.residual_quantized import ResidualQuantizedSAE
from quantized_sae.sae.baseline import BaselineSparseAutoencoder
import numpy as np

class Trainer:
    """
    Unified trainer for all SAE variants.
    """

    def __init__(self, config, sae_type, rigL=False, no_log=False, proj_name=None):
        self.config = config
        if not no_log:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"Using GPU: {torch.cuda.get_device_name(0)}")
            else:
                self.device = torch.device("cpu")
                print("GPU not available, using CPU")
        else:
            self.device = "cpu"

        self.sae_type = sae_type

        # Initialize the appropriate SAE model based on type
        if sae_type == "baseline_sae":
            self.model = BaselineSparseAutoencoder(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "t_sae":
            self.model = TernarySparseAutoencoder(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "bl_sae":
            self.model = BinaryLatentSAE(self.config["input_dim"], self.config["hidden_dim"]).to(self.device)
        elif sae_type == "b_sae":
            # Binary SAE with two's complement encoding - gamma parameter controls quantization range
            gamma = self.config.get("gamma", 4.0)  # Default gamma = 4.0
            self.model = BinarySAE(self.config["input_dim"], self.config["hidden_dim"], gamma, self.config["n_bits"]).to(self.device)
            # Pre-compute scale factors for binary weight interpretation (two's complement)
            self.scale_factor = torch.pow(2, torch.arange(self.config["n_bits"])).to(self.device)
            self.scale_factor = self.scale_factor / self.scale_factor.sum().float()
        elif sae_type == "q_sae":
            # Quantized Matryoshka SAE - hierarchical quantization with nested dictionaries
            abs_range = self.config.get("gamma", 4.0)  # Use gamma as abs_range
            self.model = QuantizedMatryoshkaSAE(
                self.config["input_dim"],
                self.config["hidden_dim"],
                self.config["top_k"],
                abs_range,
                self.config["n_bits"]
            ).to(self.device)
        elif sae_type == "rq_sae":
            # Residual Quantized SAE - residual connections between quantization levels
            abs_range = self.config.get("gamma", 4.0)  # Use gamma as abs_range
            self.model = ResidualQuantizedSAE(
                self.config["input_dim"],
                self.config["hidden_dim"],
                self.config["top_k"],
                abs_range,
                self.config["n_bits"]
            ).to(self.device)
        else:
            raise ValueError(f"Unknown SAE type: {sae_type}")

        self.epoch = 0

        # Setup logging
        if not no_log:
            if proj_name is None:
                proj_name = f"{sae_type}_training"
            try:
                wandb.init(project=proj_name, config=config)
                self.wandb = True
            except:
                print("Wandb not available, proceeding without logging")
                self.wandb = False
        else:
            self.wandb = False

def get_default_config(sae_type):
    """
    Get default configuration for each SAE type.
    """
    base_config = {
        "input_dim": 512,
        "hidden_dim": 32768,
        "lr": 1e-3,
        "batch_size": 4096,
        "epochs": 100,
    }

    # Add type-specific parameters
    if sae_type in ["b_sae", "q_sae", "rq_sae"]:
        base_config.update({
            "n_bits": 4,
            "gamma": 4.0,  # Quantization range parameter
        })

    if sae_type in ["q_sae", "rq_sae"]:
        base_config.update({
            "top_k": 32,  # Sparsity level
        })

    return base_config

def main():
    parser = argparse.ArgumentParser(description="Train Quantized SAE models")
    parser.add_argument("--sae_type", type=str, required=True,
                       choices=["baseline_sae", "b_sae", "t_sae", "bl_sae", "q_sae", "rq_sae"],
                       help="Type of SAE to train")
    parser.add_argument("--input_dim", type=int, default=512,
                       help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=32768,
                       help="Hidden dimension (sparsity level)")
    parser.add_argument("--n_bits", type=int, default=4,
                       help="Number of bits for quantized SAEs")
    parser.add_argument("--gamma", type=float, default=4.0,
                       help="Quantization range parameter")
    parser.add_argument("--top_k", type=int, default=32,
                       help="Sparsity level for hierarchical SAEs")
    parser.add_argument("--lr", type=float, default=1e-3,
                       help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=4096,
                       help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to training data (.pt file)")
    parser.add_argument("--no_log", action="store_true",
                       help="Disable wandb logging")
    parser.add_argument("--proj_name", type=str, default=None,
                       help="Wandb project name")

    args = parser.parse_args()

    # Build configuration
    config = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
    }

    # Add type-specific parameters
    if args.sae_type in ["b_sae", "q_sae", "rq_sae"]:
        config.update({
            "n_bits": args.n_bits,
            "gamma": args.gamma,
        })

    if args.sae_type in ["q_sae", "rq_sae"]:
        config.update({
            "top_k": args.top_k,
        })

    print(f"Training {args.sae_type} with config:")
    for k, v in config.items():
        print(f"  {k}: {v}")

    # Initialize trainer
    trainer = Trainer(config, args.sae_type, no_log=args.no_log, proj_name=args.proj_name)

    # Load dataset
    print(f"Loading dataset from {args.data_path}")
    dataset = HiddenStatesTorchDataset(args.data_path)
    dataloader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Training logic would go here (copied from original trainer)
    print("Training logic implementation would follow...")
    print("This is a template - actual training implementation needs to be added")

if __name__ == "__main__":
    main()