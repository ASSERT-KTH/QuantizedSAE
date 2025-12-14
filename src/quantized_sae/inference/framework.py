from dataclasses import dataclass
from pathlib import Path
from collections import OrderedDict
from typing import Any, Callable, Dict, Iterable, Iterator, Optional, Union

import torch
import torch.nn as nn

from SAEs.baseline_SAE import BaselineSparseAutoencoder
from SAEs.binary_SAE import BinarySAE
from SAEs.quantized_matryoshka_SAE import QuantizedMatryoshkaSAE
from SAEs.residual_quantized_matryoshka_SAE import ResidualQuantizedSAE

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:  # pragma: no cover - optional dependency
    load_safetensors = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Registry metadata


_MODULE_ROOT = Path(__file__).resolve().parent
_TRAINED_ROOT = _MODULE_ROOT / "Trained_SAEs"
_BASELINE_CHECKPOINT = (
    _MODULE_ROOT
    / "baseline_SAE"
    / "EleutherAI"
    / "sae-pythia-70m-32k-dedup"
    / "layers.3"
    / "sae.safetensors"
)


DeviceLike = Union[torch.device, str]


def _default_device(device: Optional[DeviceLike]) -> torch.device:
    if device is not None:
        return torch.device(device)
    return torch.device("cpu")


def _detach_to_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().cpu().clone()


def _ensure_tensor(batch: Any) -> torch.Tensor:
    if isinstance(batch, (list, tuple)):
        if not batch:
            raise ValueError("Received an empty batch; cannot infer tensor input.")
        batch = batch[0]
    if not isinstance(batch, torch.Tensor):
        raise TypeError(
            f"Expected batch to be a torch.Tensor, received {type(batch)} instead."
        )
    return batch


ForwardAdapter = Callable[[nn.Module, torch.Tensor], Dict[str, Any]]
DecoderExtractor = Callable[[nn.Module, Dict[str, Any]], Dict[str, torch.Tensor]]


@dataclass(frozen=True)
class SAERegistryEntry:
    name: str
    constructor: Callable[..., nn.Module]
    checkpoint_path: Path
    checkpoint_format: str  # "torch" | "safetensors"
    kwargs: Dict[str, Any]
    forward_adapter: ForwardAdapter
    decoder_getter: DecoderExtractor


def _forward_binary(model: BinarySAE, batch: torch.Tensor) -> Dict[str, Any]:
    sparse_latent, reconstruction, polarize_loss = model(batch)
    return {
        "latent": sparse_latent,
        "reconstruction": reconstruction,
        "aux": {"polarize_loss": polarize_loss},
    }


def _forward_quantized(
    model: QuantizedMatryoshkaSAE, batch: torch.Tensor
) -> Dict[str, Any]:
    latent_groups, reconstruction_levels = model(batch)
    return {
        "latent_groups": latent_groups,
        "reconstruction_levels": reconstruction_levels,
        "reconstruction": reconstruction_levels[-1],
    }


def _forward_residual(
    model: ResidualQuantizedSAE, batch: torch.Tensor
) -> Dict[str, Any]:
    latent_groups, reconstruction_levels = model(batch)
    return {
        "latent_groups": latent_groups,
        "reconstruction_levels": reconstruction_levels,
        "reconstruction": reconstruction_levels[-1],
    }


def _forward_baseline(
    model: BaselineSparseAutoencoder, batch: torch.Tensor
) -> Dict[str, Any]:
    latent, reconstruction = model(batch)
    return {"latent": latent, "reconstruction": reconstruction}


def _decoder_binary(
    model: BinarySAE, options: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    decoder = model.decoder
    with torch.no_grad():
        int_weights = decoder.quantized_int_weights().to(torch.float32)
        effective_weight = decoder.quantization_step * int_weights
    return {
        "weight": _detach_to_cpu(effective_weight),
        "bias": _detach_to_cpu(decoder.bias),
    }


def _decoder_quantized(
    model: QuantizedMatryoshkaSAE, _: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    decoder = model.decoder
    weight = _detach_to_cpu(decoder.weight)
    weight_mirror = _detach_to_cpu(decoder.weight_mirror)
    return {
        "weight": weight,
        "weight_mirror": weight_mirror,
        "effective_weight": weight + weight_mirror,
        "bias": _detach_to_cpu(decoder.bias),
    }


def _decoder_residual(
    model: ResidualQuantizedSAE, _: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    tensors: Dict[str, torch.Tensor] = {}
    for level, sae in enumerate(model.saes):
        weight = _detach_to_cpu(sae.decoder.weight)
        weight_mirror = _detach_to_cpu(sae.decoder.weight_mirror)
        tensors[f"level_{level}_weight"] = weight
        tensors[f"level_{level}_weight_mirror"] = weight_mirror
        tensors[f"level_{level}_effective_weight"] = weight + weight_mirror
        if hasattr(sae.decoder, "bias") and sae.decoder.bias is not None:
            tensors[f"level_{level}_bias"] = _detach_to_cpu(sae.decoder.bias)
    return tensors


def _decoder_baseline(
    model: BaselineSparseAutoencoder, _: Dict[str, Any]
) -> Dict[str, torch.Tensor]:
    return {
        "weight": _detach_to_cpu(model.decoder.weight),
        "bias": _detach_to_cpu(model.decoder.bias),
    }


SAE_REGISTRY: Dict[str, SAERegistryEntry] = {
    "b_sae": SAERegistryEntry(
        name="b_sae",
        constructor=BinarySAE,
        checkpoint_path=_TRAINED_ROOT / "b_sae_32768_4_bits.pth",
        checkpoint_format="torch",
        kwargs={
            "input_dim": 512,
            "hidden_dim": 32768,
            "gamma": 1.5,
            "n_bits": 4,
        },
        forward_adapter=_forward_binary,
        decoder_getter=_decoder_binary,
    ),
    "q_sae": SAERegistryEntry(
        name="q_sae",
        constructor=QuantizedMatryoshkaSAE,
        checkpoint_path=_TRAINED_ROOT / "q_sae_32768_4_bits.pth",
        checkpoint_format="torch",
        kwargs={
            "input_dim": 512,
            "hidden_dim": 32768,
            "top_k": 32,
            "abs_range": 1.5,
            "n_bits": 4,
            "allow_bias": True,
        },
        forward_adapter=_forward_quantized,
        decoder_getter=_decoder_quantized,
    ),
    "rq_sae": SAERegistryEntry(
        name="rq_sae",
        constructor=ResidualQuantizedSAE,
        checkpoint_path=_TRAINED_ROOT / "rq_sae_32768_4_bits.pth",
        checkpoint_format="torch",
        kwargs={
            "input_dim": 512,
            "hidden_dim": 32768,
            "top_k": 32,
            "abs_range": 1.5,
            "n_bits": 4,
        },
        forward_adapter=_forward_residual,
        decoder_getter=_decoder_residual,
    ),
    "baseline_sae": SAERegistryEntry(
        name="baseline_sae",
        constructor=BaselineSparseAutoencoder,
        checkpoint_path=_MODULE_ROOT / "SAEs" / "baseline_sae_32768.pth",
        checkpoint_format="torch",
        kwargs={"input_dim": 512, "hidden_dim": 32768},
        forward_adapter=_forward_baseline,
        decoder_getter=_decoder_baseline,
    ),
}


# ---------------------------------------------------------------------------
# Loader and wrapper


def _load_state_dict(entry: SAERegistryEntry) -> Dict[str, torch.Tensor]:
    if not entry.checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for '{entry.name}': {entry.checkpoint_path}"
        )

    if entry.checkpoint_format == "torch":
        state_dict = torch.load(entry.checkpoint_path, map_location="cpu")
    elif entry.checkpoint_format == "safetensors":
        if load_safetensors is None:
            try:
                # Local pure-python fallback that does not require the safetensors package.
                from load_baseline_sae import load_safetensors as local_loader  # type: ignore
            except ImportError:
                try:
                    from customSAE.load_baseline_sae import (  # type: ignore
                        load_safetensors as local_loader,
                    )
                except ImportError as exc:  # pragma: no cover - defensive branch
                    raise ImportError(
                        "safetensors is required to load baseline SAE checkpoints "
                        "and the fallback loader `customSAE.load_baseline_sae` was "
                        "not found. Install safetensors with "
                        "`pip install safetensors` or ensure the fallback module is available."
                    ) from exc
            raw_state = local_loader(str(entry.checkpoint_path))
            state_dict = OrderedDict(
                {
                    "encoder.0.weight": raw_state["encoder.weight"].clone(),
                    "encoder.0.bias": raw_state["encoder.bias"].clone(),
                    "decoder.weight": raw_state["W_dec"].clone().t().contiguous(),
                    "decoder.bias": raw_state["b_dec"].clone(),
                }
            )
        else:
            state_dict = load_safetensors(str(entry.checkpoint_path))
            if "encoder.0.weight" not in state_dict:
                state_dict = OrderedDict(
                    {
                        "encoder.0.weight": state_dict["encoder.weight"],
                        "encoder.0.bias": state_dict["encoder.bias"],
                        "decoder.weight": state_dict["W_dec"].t().contiguous(),
                        "decoder.bias": state_dict["b_dec"],
                    }
                )
    else:  # pragma: no cover - defensive branch
        raise ValueError(
            f"Unsupported checkpoint format '{entry.checkpoint_format}' "
            f"for SAE '{entry.name}'."
        )
    return state_dict


class SAEWrapper:
    """Unified interface for running inference with heterogeneous SAE variants.

    Examples
    --------
    >>> from torch.utils.data import DataLoader, TensorDataset
    >>> loader = DataLoader(TensorDataset(torch.randn(8, 512)), batch_size=2)
    >>> sae = load_sae(\"b_sae\")  # Automatically loads weights onto GPU if available.
    >>> for batch in loader:
    ...     outputs = sae(batch[0])
    ...     recon = outputs[\"reconstruction\"]
    ...     # Process reconstructions here
    >>> decoder = sae.decoder_dictionary(quantized=True)  # Access decoder weights.
    """

    def __init__(
        self,
        entry: SAERegistryEntry,
        model: nn.Module,
        device: Optional[DeviceLike],
    ) -> None:
        self._entry = entry
        self.model = model
        self.device = _default_device(device)
        self.model.to(self.device)
        self.model.eval()

    def to(self, device: DeviceLike) -> "SAEWrapper":
        self.device = torch.device(device)
        self.model.to(self.device)
        return self

    def eval(self) -> "SAEWrapper":
        self.model.eval()
        return self

    @torch.no_grad()
    def __call__(self, batch: torch.Tensor) -> Dict[str, Any]:
        batch = _ensure_tensor(batch).to(self.device)
        return self._entry.forward_adapter(self.model, batch)

    @torch.no_grad()
    def reconstruct(self, batch: torch.Tensor) -> torch.Tensor:
        return self(batch)["reconstruction"]

    @torch.no_grad()
    def reconstruct_loader(
        self,
        dataloader: Iterable[Any],
        *,
        return_details: bool = False,
    ) -> Iterator[Union[torch.Tensor, Dict[str, Any]]]:
        for batch in dataloader:
            outputs = self(batch)
            yield outputs if return_details else outputs["reconstruction"]

    def decoder_dictionary(self, **options: Any) -> Dict[str, torch.Tensor]:
        return self._entry.decoder_getter(self.model, options)


def available_saes() -> Dict[str, Path]:
    """Return mapping of model keys to their checkpoint paths."""
    return {name: entry.checkpoint_path for name, entry in SAE_REGISTRY.items()}


def load_sae(
    name: str,
    *,
    device: Optional[DeviceLike] = None,
    strict: bool = True,
) -> SAEWrapper:
    """Instantiate an SAE variant, restore weights, and wrap it for inference."""
    if name not in SAE_REGISTRY:
        raise KeyError(f"Unknown SAE '{name}'. Available: {list(SAE_REGISTRY)}")

    entry = SAE_REGISTRY[name]
    state_dict = _load_state_dict(entry)
    model = entry.constructor(**entry.kwargs)
    model.load_state_dict(state_dict, strict=strict)
    return SAEWrapper(entry, model, device)


