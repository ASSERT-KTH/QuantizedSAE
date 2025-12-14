#!/usr/bin/env python3
"""
Estimate the quantization error introduced when converting a trained BinarySAE
decoder's probability-weighted weights into hard integer weights via
``binary_decoder.quantized_int_weights``.

The script loads a checkpoint, reconstructs:
  * ``W_float``  – the continuous decoder weights used during training
  * ``W_quant``  – the hard-quantized integer weights scaled by the decoder's step
and then reports summary statistics of ``W_quant - W_float``.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch

# Ensure local imports are resolvable when launched from arbitrary directories.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from SAEs.binary_SAE import BinarySAE, binary_decoder  # noqa: E402


def _resolve_state_dict(raw_checkpoint: object) -> Dict[str, torch.Tensor]:
    """
    Extract a ``state_dict`` from a checkpoint saved in a variety of common layouts.
    """

    if isinstance(raw_checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            value = raw_checkpoint.get(key)
            if isinstance(value, dict):
                return value
        return raw_checkpoint  # assume dict already is the state_dict
    return raw_checkpoint  # type: ignore[return-value]


def _infer_model_config(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int, int]:
    """
    Infer ``hidden_dim``, ``input_dim``, and ``n_bits`` from checkpoint tensors.

    Returns:
        Tuple[int, int, int]: (hidden_dim, input_dim, n_bits)
    """

    encoder_weight_keys = ("encoder.0.weight", "encoder.weight")
    decoder_weight_keys = ("decoder.weight",)

    encoder_weight = None
    for key in encoder_weight_keys:
        if key in state_dict:
            encoder_weight = state_dict[key]
            break
    if encoder_weight is None:
        raise KeyError("Unable to locate encoder weight in checkpoint for dimension inference.")

    decoder_weight = None
    for key in decoder_weight_keys:
        if key in state_dict:
            decoder_weight = state_dict[key]
            break
    if decoder_weight is None:
        raise KeyError("Unable to locate decoder weight in checkpoint for dimension inference.")

    hidden_dim, input_dim = encoder_weight.shape
    decoder_cols = decoder_weight.shape[1]
    if decoder_cols % input_dim != 0:
        raise ValueError(
            f"Decoder weight shape mismatch: expected columns to be a multiple of input_dim "
            f"({input_dim}), but got {decoder_cols}."
        )
    n_bits = decoder_cols // input_dim
    return hidden_dim, input_dim, n_bits


def load_model(
    model_path: Path,
    input_dim: Optional[int],
    hidden_dim: Optional[int],
    n_bits: Optional[int],
    gamma: float,
    device: torch.device,
) -> BinarySAE:
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = _resolve_state_dict(checkpoint)

    inferred_hidden_dim, inferred_input_dim, inferred_n_bits = _infer_model_config(state_dict)

    final_hidden_dim = inferred_hidden_dim if hidden_dim is None else hidden_dim
    if final_hidden_dim != inferred_hidden_dim:
        print(
            f"[INFO] Overriding provided hidden_dim={final_hidden_dim} with "
            f"checkpoint value {inferred_hidden_dim}.",
            file=sys.stderr,
        )
        final_hidden_dim = inferred_hidden_dim

    final_input_dim = inferred_input_dim if input_dim is None else input_dim
    if final_input_dim != inferred_input_dim:
        print(
            f"[INFO] Overriding provided input_dim={final_input_dim} with "
            f"checkpoint value {inferred_input_dim}.",
            file=sys.stderr,
        )
        final_input_dim = inferred_input_dim

    final_n_bits = inferred_n_bits if n_bits is None else n_bits
    if final_n_bits != inferred_n_bits:
        print(
            f"[INFO] Overriding provided n_bits={final_n_bits} with checkpoint value {inferred_n_bits}.",
            file=sys.stderr,
        )
        final_n_bits = inferred_n_bits

    model = BinarySAE(final_input_dim, final_hidden_dim, gamma=gamma, n_bits=final_n_bits)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def recover_float_decoder(model: BinarySAE) -> torch.Tensor:
    """
    Reconstruct the continuous decoder weight matrix used during training.
    Shape: [hidden_dim, input_dim]
    """

    decoder = model.decoder
    prob_weights = torch.sigmoid(decoder.weight)
    prob_weights = prob_weights.view(decoder.in_features, -1, decoder.n_bits)

    bit_weights = torch.pow(
        torch.tensor(2.0, device=prob_weights.device, dtype=prob_weights.dtype),
        torch.arange(decoder.n_bits, device=prob_weights.device, dtype=prob_weights.dtype),
    )
    bit_weights[-1] *= -1

    int_expectation = (prob_weights * bit_weights).sum(dim=-1)
    return decoder.quantization_step * int_expectation


@torch.no_grad()
def recover_quantized_decoder(model: BinarySAE) -> torch.Tensor:
    """
    Recover the hard-quantized decoder weights (scaled to real values).
    Shape: [hidden_dim, input_dim]
    """

    decoder = model.decoder
    int_weights = decoder.quantized_int_weights().to(decoder.bias.device)
    return decoder.quantization_step * int_weights.float()


def summarize_error(diff: torch.Tensor) -> Dict[str, float]:
    mse = diff.pow(2).mean().item()
    mean_abs = diff.abs().mean().item()
    max_abs = diff.abs().max().item()
    l2_norm = diff.norm().item()
    return {
        "mse": mse,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "l2_norm": l2_norm,
    }


def summarize_matrix(matrix: torch.Tensor, prefix: str) -> Dict[str, float]:
    return {
        f"{prefix}_mean": matrix.mean().item(),
        f"{prefix}_std": matrix.std(unbiased=False).item(),
        f"{prefix}_min": matrix.min().item(),
        f"{prefix}_max": matrix.max().item(),
        f"{prefix}_l2_norm": matrix.norm().item(),
    }


def collect_bit_details(decoder: binary_decoder, row_idx: int, col_idx: int) -> Tuple[Dict[str, float], ...]:
    """Gather per-bit statistics for a specific decoder weight."""

    n_bits = decoder.n_bits
    start = col_idx * n_bits
    end = start + n_bits

    logits = decoder.weight[row_idx, start:end]
    prob_bits = torch.sigmoid(logits)
    hard_bits = (prob_bits > 0.5).to(logits.dtype)

    bit_weights = torch.pow(
        torch.tensor(2.0, device=prob_bits.device, dtype=prob_bits.dtype),
        torch.arange(n_bits, device=prob_bits.device, dtype=prob_bits.dtype),
    )
    bit_weights[-1] *= -1

    quant_step = decoder.quantization_step

    float_contrib = (prob_bits * bit_weights) * quant_step
    quant_contrib = (hard_bits * bit_weights) * quant_step

    details = []
    for bit_idx in range(n_bits):
        details.append(
            {
                "bit_index": bit_idx,
                "logit": logits[bit_idx].item(),
                "prob": prob_bits[bit_idx].item(),
                "hard": int(hard_bits[bit_idx].item()),
                "bit_weight": bit_weights[bit_idx].item(),
                "float_contrib": float_contrib[bit_idx].item(),
                "quant_contrib": quant_contrib[bit_idx].item(),
            }
        )
    return tuple(details)


def find_max_diff_entry(
    model: BinarySAE, w_float: torch.Tensor, w_quant: torch.Tensor
) -> Dict[str, float]:
    """
    Identify the index with the largest absolute difference and record the
    associated values from both matrices.
    """

    diff = (w_quant - w_float).abs()
    max_val, max_idx_tensor = diff.view(-1).max(0)
    max_idx = int(max_idx_tensor.item())

    rows, cols = w_float.shape
    row_idx = max_idx // cols
    col_idx = max_idx % cols

    w_float_val = w_float[row_idx, col_idx].item()
    w_quant_val = w_quant[row_idx, col_idx].item()
    signed_diff = w_quant_val - w_float_val

    bit_details = collect_bit_details(model.decoder, row_idx, col_idx)

    return {
        "row_index": row_idx,
        "col_index": col_idx,
        "w_float_value": w_float_val,
        "w_quant_value": w_quant_val,
        "signed_diff": signed_diff,
        "abs_diff": max_val.item(),
        "bit_details": bit_details,
    }


def format_report(stats: Dict[str, float], max_diff_info: Dict[str, float]) -> str:
    lines = [
        "=== Decoder Weight Quantization Report ===",
        f"MSE(W_quant - W_float):    {stats['mse']:.6e}",
        f"Mean |ΔW|:                 {stats['mean_abs']:.6e}",
        f"Max  |ΔW|:                 {stats['max_abs']:.6e}",
        f"L2  ||ΔW||:                {stats['l2_norm']:.6e}",
        "",
        f"W_float mean/std/min/max: ({stats['float_mean']:.6e}, {stats['float_std']:.6e}, "
        f"{stats['float_min']:.6e}, {stats['float_max']:.6e})",
        f"W_quant mean/std/min/max: ({stats['quant_mean']:.6e}, {stats['quant_std']:.6e}, "
        f"{stats['quant_min']:.6e}, {stats['quant_max']:.6e})",
        "",
        f"||W_float||_2: {stats['float_l2_norm']:.6e}",
        f"||W_quant||_2: {stats['quant_l2_norm']:.6e}",
        "",
        "Largest absolute difference entry:",
        f"  Indices (hidden, input): ({max_diff_info['row_index']}, {max_diff_info['col_index']})",
        f"  W_float value:           {max_diff_info['w_float_value']:.6e}",
        f"  W_quant value:           {max_diff_info['w_quant_value']:.6e}",
        f"  Signed diff:             {max_diff_info['signed_diff']:.6e}",
        f"  Absolute diff:           {max_diff_info['abs_diff']:.6e}",
    ]

    bit_lines = ["", "  Bit-level details (index, logit, prob, hard, bit_weight, float_contrib, quant_contrib):"]
    for bit in max_diff_info["bit_details"]:
        bit_lines.append(
            "    "
            f"bit {bit['bit_index']}: "
            f"logit={bit['logit']:.6e}, "
            f"prob={bit['prob']:.6e}, "
            f"hard={bit['hard']}, "
            f"weight={bit['bit_weight']:.6e}, "
            f"float={bit['float_contrib']:.6e}, "
            f"quant={bit['quant_contrib']:.6e}"
        )

    lines.extend(bit_lines)
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compute decoder weight quantization statistics for a BinarySAE checkpoint."
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Path to the BinarySAE checkpoint (e.g. Trained_SAEs/b_sae_32768_4_bits.pth).",
    )
    parser.add_argument(
        "--input-dim",
        type=int,
        default=None,
        help="Input dimension used during training. Defaults to checkpoint value.",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=None,
        help="Hidden dimension of the BinarySAE. Defaults to checkpoint value.",
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=None,
        help="Number of bits per decoder entry. Defaults to checkpoint value.",
    )
    parser.add_argument(
        "--gamma", type=float, default=4.0, help="Decoder scaling hyper-parameter (sets quantization step size)."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Computation device (defaults to CUDA if available).",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_model(
        model_path=args.model_path,
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        n_bits=args.n_bits,
        gamma=args.gamma,
        device=device,
    )

    w_float = recover_float_decoder(model)
    w_quant = recover_quantized_decoder(model)
    diff = w_quant - w_float

    stats = summarize_error(diff)
    stats.update(summarize_matrix(w_float, "float"))
    stats.update(summarize_matrix(w_quant, "quant"))

    max_diff_info = find_max_diff_entry(model, w_float, w_quant)

    print(format_report(stats, max_diff_info))


if __name__ == "__main__":
    main()

