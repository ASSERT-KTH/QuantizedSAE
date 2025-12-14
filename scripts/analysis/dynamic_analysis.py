import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except ImportError:  # fallback: no-op progress bar
    def tqdm(iterable, *args, **kwargs):
        return iterable

from sae_inference_framework import SAEWrapper
from SAEs.binary_SAE import BinarySAE
from SAEs.quantized_matryoshka_SAE import QuantizedMatryoshkaSAE
from SAEs.residual_quantized_matryoshka_SAE import ResidualQuantizedSAE
from SAEs.baseline_SAE import BaselineSparseAutoencoder


def _hidden_dim(sae: SAEWrapper) -> int:
    model = sae.model
    if hasattr(model, "hidden_dim"):
        return int(model.hidden_dim)
    if isinstance(model, BaselineSparseAutoencoder):
        # decoder: [input_dim, hidden_dim]
        return int(model.decoder.weight.shape[1])
    if isinstance(model, ResidualQuantizedSAE):
        return int(sum(s.hidden_dim for s in model.saes))
    raise ValueError(f"Unable to determine hidden_dim for model type {type(model)}")


def _activation_mask(sae: SAEWrapper, x: torch.Tensor) -> torch.Tensor:
    """
    Boolean mask [batch, hidden_dim] indicating which features are active.

    - BinarySAE: latent after masking > 0
    - Baseline: encoder output (ReLU) > 0
    - QuantizedMatryoshkaSAE: encoder output (sigmoid) > 0.5
    - ResidualQuantizedSAE: concatenation of per-level encoder(residual) > 0.5,
      with residual updated exactly as in the forward pass.
    """
    model = sae.model
    with torch.no_grad():
        if isinstance(model, BinarySAE):
            latent, _, _ = model(x)
            mask = latent > 0

        elif isinstance(model, BaselineSparseAutoencoder):
            h, _ = model(x)
            mask = h > 0

        elif isinstance(model, QuantizedMatryoshkaSAE):
            latent = model.encoder(x)
            mask = latent > 0.5

        elif isinstance(model, ResidualQuantizedSAE):
            B = x.size(0)
            H = _hidden_dim(sae)
            mask = torch.zeros(B, H, dtype=torch.bool, device=x.device)

            residual = x
            start = 0
            for sub_sae, size in zip(model.saes, model.sae_hidden_dims):
                latent = sub_sae.encoder(residual)
                mask[:, start : start + size] = latent > 0.5

                # Forward one residual step with same logic as ResidualQuantizedSAE.forward
                _, recon_levels = sub_sae.decoder(latent)
                reconstruction = recon_levels[-1]
                residual = (residual - reconstruction).detach() * 2
                start += size
        else:
            raise TypeError(f"Unsupported SAE model type: {type(model)}")

    return mask.cpu()


def compute_reconstruction_error(
    sae: SAEWrapper,
    loader: DataLoader,
    device: str = "cpu",
) -> float:
    """
    Unbiased reconstruction MSE over the dataset:
    mean of (recon - x)^2 across all tokens and dimensions.
    """
    sae.to(device).eval()
    sq_err_sum = 0.0
    n_elements = 0

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)

            recon = sae.reconstruct(batch)
            diff = recon - batch
            sq_err_sum += (diff * diff).sum().item()
            n_elements += diff.numel()

    return sq_err_sum / n_elements


def compute_reconstruction_error_by_level(
    sae: SAEWrapper,
    loader: DataLoader,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Per-level reconstruction MSE for architectures that expose reconstruction groups:

    - QuantizedMatryoshkaSAE (q_sae):
        uses all reconstructions in the group, loss vs original x.
    - ResidualQuantizedSAE (rq_sae):
        uses each level's reconstruction vs the current residual
        (matching the training objective).

    For other SAEs, this falls back to a single-element tensor containing
    the overall reconstruction error.
    """
    sae.to(device).eval()
    model = sae.model

    # Binary / baseline: just the global MSE as length-1 tensor
    if not isinstance(model, (QuantizedMatryoshkaSAE, ResidualQuantizedSAE)):
        mse = compute_reconstruction_error(sae, loader, device=device)
        return torch.tensor([mse], dtype=torch.float64)

    sum_sqs = None
    counts = None

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            batch = batch.to(device)

            if isinstance(model, QuantizedMatryoshkaSAE):
                # model(batch) -> (latent_groups, reconstruction_levels)
                _, recon_levels = model(batch)
                n_levels = len(recon_levels)
                if sum_sqs is None:
                    sum_sqs = torch.zeros(n_levels, dtype=torch.float64)
                    counts = torch.zeros(n_levels, dtype=torch.float64)

                for i, recon in enumerate(recon_levels):
                    diff = recon - batch
                    sum_sqs[i] += (diff * diff).sum().item()
                    counts[i] += diff.numel()

            elif isinstance(model, ResidualQuantizedSAE):
                # model(batch) -> (all_latent_groups, all_reconstruction_levels)
                _, recon_levels = model(batch)
                n_levels = len(recon_levels)
                if sum_sqs is None:
                    sum_sqs = torch.zeros(n_levels, dtype=torch.float64)
                    counts = torch.zeros(n_levels, dtype=torch.float64)

                residual = batch
                for i, recon in enumerate(recon_levels):
                    diff = recon - residual
                    sum_sqs[i] += (diff * diff).sum().item()
                    counts[i] += diff.numel()
                    residual = (residual - recon).detach() * 2

    return sum_sqs / counts


def compute_l0_by_level(
    sae: SAEWrapper,
    loader: DataLoader,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Average L0 (number of active units) per token for each level.

    - For QuantizedMatryoshkaSAE (q_sae):
        Levels are the nested dictionary slices given by
        decoder.nested_dictionary_size. We binarize encoder(x) at 0.5.

    - For ResidualQuantizedSAE (rq_sae):
        Levels are the residual stages (each inner QuantizedMatryoshkaSAE
        with n_bits=1). We binarize each stage's encoder(residual) at 0.5,
        updating residual as in the forward pass.

    For other SAEs, returns a length-1 tensor with the global average L0.
    """
    sae.to(device).eval()
    model = sae.model

    # Binary / baseline: single level equal to total active count per token
    if not isinstance(model, (QuantizedMatryoshkaSAE, ResidualQuantizedSAE)):
        total_act = 0.0
        n_tokens = 0.0
        with torch.no_grad():
            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                mask = _activation_mask(sae, batch)  # [B, H]
                total_act += mask.sum().item()
                n_tokens += mask.size(0)
        return torch.tensor([total_act / max(n_tokens, 1.0)], dtype=torch.float64)

    with torch.no_grad():
        if isinstance(model, QuantizedMatryoshkaSAE):
            sizes = list(model.decoder.nested_dictionary_size)
            n_levels = len(sizes)
            total_act = torch.zeros(n_levels, dtype=torch.float64)
            n_tokens = 0.0

            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                latent = model.encoder(batch)               # [B, H], sigmoid
                latent_bin = (latent > 0.5).to(torch.int64) # [B, H]

                start = 0
                for i, size in enumerate(sizes):
                    sl = latent_bin[:, start : start + size]
                    total_act[i] += sl.sum().item()
                    start += size
                n_tokens += latent_bin.size(0)

            return total_act / max(n_tokens, 1.0)

        else:  # ResidualQuantizedSAE
            n_levels = len(model.saes)
            total_act = torch.zeros(n_levels, dtype=torch.float64)
            n_tokens = 0.0

            for batch in loader:
                if isinstance(batch, (list, tuple)):
                    batch = batch[0]
                batch = batch.to(device)
                B = batch.size(0)
                residual = batch

                for i, sub_sae in enumerate(model.saes):
                    latent = sub_sae.encoder(residual)           # [B, h_i], sigmoid
                    latent_bin = (latent > 0.5).to(torch.int64)
                    total_act[i] += latent_bin.sum().item()

                    # Match residual update logic
                    _, recon_levels = sub_sae.decoder(latent)
                    reconstruction = recon_levels[-1]
                    residual = (residual - reconstruction).detach() * 2

                n_tokens += B

            return total_act / max(n_tokens, 1.0)


def compute_activation_stats(
    sae: SAEWrapper,
    loader: DataLoader,
    *,
    token_ids: torch.Tensor,  # [num_contexts, tokens_per_context]
    tokens_per_context: int,
    device: str = "cpu",
) -> dict:
    """
    For a given SAE, compute:
      - activation_counts: [H]      – times each feature is active
      - coactivation: [H, H]        – A^T A, co-activation counts
      - tokens_per_feature: list(H) – list of token IDs where feature was active
    """
    sae.to(device).eval()
    H = _hidden_dim(sae)

    activation_counts = torch.zeros(H, dtype=torch.long)
    coactivation = torch.zeros(H, H, dtype=torch.int32)
    tokens_per_feature = [[] for _ in range(H)]

    global_index = 0  # flattened token index

    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            B = batch.size(0)

            # Map global token indices -> (context_idx, token_idx)
            flat_idx = torch.arange(global_index, global_index + B, dtype=torch.long)
            context_idx = torch.div(flat_idx, tokens_per_context, rounding_mode="floor")
            token_idx = flat_idx % tokens_per_context
            batch_token_ids = token_ids[context_idx, token_idx]  # [B]

            batch = batch.to(device)
            mask = _activation_mask(sae, batch)  # [B, H] on CPU

            # 1) Activation counts
            activation_counts += mask.sum(dim=0).to(torch.long)

            # 2) Co-activation matrix via A^T A
            mask_int = mask.to(dtype=torch.int32)
            coactivation += mask_int.t() @ mask_int

            # 3) Tokens per feature
            nz = mask.nonzero(as_tuple=False)  # [num_acts, 2]
            if nz.numel() > 0:
                sample_idx = nz[:, 0]
                feat_idx = nz[:, 1]
                tok_vals = batch_token_ids[sample_idx]
                for f, t in zip(feat_idx.tolist(), tok_vals.tolist()):
                    tokens_per_feature[f].append(int(t))

            global_index += B

    return {
        "activation_counts": activation_counts,
        "coactivation": coactivation,
        "tokens_per_feature": tokens_per_feature,
    }


def analyze_dataset(
    sae: SAEWrapper,
    loader: DataLoader,
    *,
    token_ids: torch.Tensor,
    tokens_per_context: int,
    device: str,  # We will pass "cuda" here
) -> dict:
    sae.to(device).eval()
    model = sae.model
    H = _hidden_dim(sae)

    # Accumulators
    mse_sq_sum = 0.0
    mse_count = 0
    
    # We keep the coactivation matrix on CPU RAM because 32k*32k int32 takes ~4GB.
    # We will compute chunks on GPU and move them to CPU to add.
    activation_counts = torch.zeros(H, dtype=torch.long, device="cpu")
    coactivation = torch.zeros((H, H), dtype=torch.int32, device="cpu")
    
    # Note: collecting tokens per feature is inherently slow/memory heavy. 
    # Only do this if you strictly need it.
    tokens_per_feature = [[] for _ in range(H)]

    global_index = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Analyzing {type(model).__name__}"):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            
            print(f"Global index: {global_index}")
            
            # 1. Move Data to GPU
            batch = batch.to(device)
            B = batch.size(0)

            # 2. Run Forward Pass
            outputs = sae(batch)
            recon_final = outputs["reconstruction"]

            # 3. Calculate Global MSE (Vectorized on GPU)
            diff_final = recon_final - batch
            mse_sq_sum += (diff_final * diff_final).sum().item()
            mse_count += diff_final.numel()

            # 4. Get Activation Mask (Vectorized)
            # We need a unified way to get the binary mask [B, H] on GPU
            if isinstance(model, QuantizedMatryoshkaSAE):
                latent = model.encoder(batch)
                batch_mask = (latent > 0.5) # Keep on GPU
            elif isinstance(model, ResidualQuantizedSAE):
                # Construct mask efficiently without moving back and forth
                batch_mask = torch.zeros(B, H, dtype=torch.bool, device=device)
                residual = batch
                start = 0
                for sub_sae, size, recon in zip(model.saes, model.sae_hidden_dims, outputs["reconstruction_levels"]):
                    latent = sub_sae.encoder(residual)
                    batch_mask[:, start : start + size] = (latent > 0.5)
                    residual = (residual - recon).detach() * 2
                    start += size
            else:
                # Binary/Baseline
                # Use your existing logic, but ensure it returns GPU tensor
                batch_mask = _activation_mask(sae, batch).to(device)

            # 5. Calculate Co-occurrence (Vectorized Matrix Multiplication)
            # 
            # Instead of a loop, we do (H, B) @ (B, H) -> (H, H)
            # We convert boolean to float for matmul, then back to CPU to save VRAM
            
            # Optimization: batch_mask is usually sparse-ish, but dense matmul is often faster 
            # than sparse overhead for these dimensions unless extremely sparse.
            batch_mask_float = batch_mask.float()
            
            # This single line replaces your loop over B
            batch_cooc = torch.matmul(batch_mask_float.T, batch_mask_float)
            
            # Move result to CPU and accumulate
            coactivation += batch_cooc.cpu().to(torch.int32)
            activation_counts += batch_mask.sum(dim=0).cpu().to(torch.long)

            # 6. Token Logging (The only remaining slow part)
            # We perform this on CPU to avoid GPU synchronization stalls
            batch_mask_cpu = batch_mask.cpu()
            
            # Calculate token IDs for this batch
            flat_idx = torch.arange(global_index, global_index + B, dtype=torch.long)
            context_idx = flat_idx // tokens_per_context
            token_idx = flat_idx % tokens_per_context
            batch_token_vals = token_ids[context_idx, token_idx] # [B]

            # Use nonzero() to find active indices rapidly
            nz = batch_mask_cpu.nonzero(as_tuple=False) # [N_activations, 2]
            if nz.numel() > 0:
                # nz[:, 0] is batch index, nz[:, 1] is feature index
                # We can't fully vectorize appending to lists, but this is cleaner
                sample_indices = nz[:, 0]
                feature_indices = nz[:, 1]
                
                # Get the actual token ID for every activation
                active_token_ids = batch_token_vals[sample_indices]
                
                # We still must loop to append to lists, but we loop over *activations*, 
                # not the strict grid. This is the unavoidable Python overhead.
                for i in range(len(feature_indices)):
                    feat = feature_indices[i].item()
                    tok = active_token_ids[i].item()
                    tokens_per_feature[feat].append(tok)

            global_index += B

    # Final Stats
    mse_final = mse_sq_sum / max(mse_count, 1)
    
    return {
        "mse_final": mse_final,
        "mse_per_level": None, # (Add back if you need strict parity with your old code)
        "l0_per_level": None,
        "activation_counts": activation_counts,
        "coactivation": coactivation,
        "tokens_per_feature": tokens_per_feature,
    }