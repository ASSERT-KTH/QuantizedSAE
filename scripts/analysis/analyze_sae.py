import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import torch
import torch.nn.functional as F

from quantized_sae.inference.framework import SAEWrapper, load_sae


def _input_dim(sae: SAEWrapper) -> int:
    if hasattr(sae.model, "input_dim"):
        return int(sae.model.input_dim)
    decoder_module = getattr(sae.model, "decoder", None)
    if decoder_module is not None and getattr(decoder_module, "bias", None) is not None:
        return int(decoder_module.bias.numel())
    if hasattr(sae.model, "saes") and len(sae.model.saes) > 0:
        first_decoder = getattr(sae.model.saes[0], "decoder", None)
        if first_decoder is not None and getattr(first_decoder, "bias", None) is not None:
            return int(first_decoder.bias.numel())
    raise ValueError("Unable to determine input dimension for SAE.")


def _decoder_features(sae: SAEWrapper) -> torch.Tensor:
    """Return decoder atoms with shape [n_features, feature_dim] on CPU."""
    decoder = sae.decoder_dictionary()
    weight_tensors = [
        tensor
        for key, tensor in decoder.items()
        if key.endswith("effective_weight") or key == "effective_weight"
    ]
    if not weight_tensors:
        if "weight" in decoder:
            weight_tensors = [decoder["weight"]]
        else:
            weight_tensors = [
                tensor
                for key, tensor in decoder.items()
                if key.endswith("_weight") and "mirror" not in key
            ]
    if not weight_tensors:
        raise ValueError("No decoder weight tensors found.")

    input_dim = _input_dim(sae)
    features = []
    for tensor in weight_tensors:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        tensor = tensor.to(dtype=torch.float32)
        if tensor.shape[1] == input_dim:
            features.append(tensor)
        elif tensor.shape[0] == input_dim:
            features.append(tensor.t())
        else:
            features.append(tensor)
    return torch.cat(features, dim=0)


def decoder_cosine_similarity(
    lhs: SAEWrapper, rhs: SAEWrapper
) -> torch.Tensor:
    """Compute pairwise cosine similarities between decoder atoms of two SAEs."""
    lhs_atoms = _decoder_features(lhs)
    rhs_atoms = _decoder_features(rhs)

    lhs_norm = F.normalize(lhs_atoms, dim=1)
    rhs_norm = F.normalize(rhs_atoms, dim=1)

    return lhs_norm @ rhs_norm.t()


if __name__ == "__main__":
    binary = load_sae("b_sae")
    quantized = load_sae("q_sae")
    residual = load_sae("rq_sae")
    baseline = load_sae("baseline_sae")

    similarity_matrix = decoder_cosine_similarity(binary, baseline)
    
    # 1. Average cosine similarity
    avg_cosine_sim = similarity_matrix.mean().item()
    print(f"Average cosine similarity: {avg_cosine_sim:.6f}")
    
    # 2. Mean of the top 100 cosine similarities for each lhs feature
    # For each feature in lhs (each row), get top 100 cosine similarities with rhs features
    num_rhs_features = similarity_matrix.shape[1]
    k = min(100, num_rhs_features)
    # Get top k cosine similarities for each lhs feature (each row)
    top_k_per_feature = torch.topk(similarity_matrix.max(dim=1).values, k=k)
    # Take mean of top k for each feature, then mean across all features
    mean_top_k_per_feature = top_k_per_feature.values.mean().item()
    print(f"Mean of top {k} cosine similarities per feature (averaged across all features): {mean_top_k_per_feature:.6f}")