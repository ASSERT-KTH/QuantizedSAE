import argparse
from pathlib import Path
from itertools import combinations
from heapq import nlargest

from collections import Counter

import torch

from sae_inference_framework import load_sae
from SAEs.quantized_matryoshka_SAE import QuantizedMatryoshkaSAE
from SAEs.residual_quantized_matryoshka_SAE import ResidualQuantizedSAE


def load_stats(root: Path, name: str) -> dict:
    path = root / f"dynamic_stats_{name}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Stats file not found for '{name}': {path}")
    return torch.load(path, map_location="cpu")


def summarize_activation_counts(activation_counts: torch.Tensor) -> float:
    """Average activation count per feature."""
    return float(activation_counts.float().mean().item())


def count_below_threshold(
    activation_counts: torch.Tensor, threshold: int
) -> int:
    """Number of features whose activation count is below the given threshold."""
    if activation_counts.numel() == 0:
        return 0
    below = (activation_counts < threshold).sum().item()
    return int(below)


def average_coactivating_features(
    coactivation: torch.Tensor,
    activation_counts: torch.Tensor,
    *,
    row_mask: torch.Tensor | None = None,
) -> float:
    """
    For each feature i, count how many distinct features j (j != i)
    co-activate with it at least once (coactivation[i, j] > 0), then
    average this count over:
      - all activated features (activation_count > 0) when row_mask is None
      - only the activated features selected by row_mask when provided
    """
    if coactivation.numel() == 0 or activation_counts.numel() == 0:
        return 0.0

    # Only consider features that were activated at least once.
    active_mask = activation_counts > 0
    if row_mask is not None:
        # Restrict rows being averaged to the requested subset.
        active_mask = active_mask & row_mask
    if not active_mask.any():
        return 0.0

    # Zero out diagonal so we do not count self-activation.
    co = coactivation.clone()
    idx = torch.arange(co.size(0))
    co[idx, idx] = 0

    # Count how many other features have positive co-activation with each feature.
    per_feature_counts = (co > 0).sum(dim=1).float()

    # Average only over activated features.
    return float(per_feature_counts[active_mask].mean().item())


def average_unique_tokens_per_active_feature(
    tokens_per_feature: list, activation_counts: torch.Tensor
) -> float:
    """
    Mean number of unique tokens for features that activated at least once.
    tokens_per_feature[i] is a list of token IDs where feature i was active.
    """
    if tokens_per_feature is None or activation_counts.numel() == 0:
        return 0.0

    active_mask = activation_counts > 0
    if not active_mask.any():
        return 0.0

    unique_counts = []
    for is_active, tok_list in zip(active_mask.tolist(), tokens_per_feature):
        if not is_active:
            continue
        # tok_list might be empty; set() handles that.
        unique_counts.append(len(set(tok_list)))

    if len(unique_counts) == 0:
        return 0.0

    return float(sum(unique_counts) / len(unique_counts))


def _topk_token_set(token_list: list, k: int) -> set[int]:
    if not token_list or k <= 0:
        return set()
    counter = Counter(token_list)
    # Keep only token IDs (discard counts), respecting top-k by frequency.
    return {tok for tok, _ in counter.most_common(k)}


def jaccard_between_saes(
    stats_a: dict, stats_b: dict, k_tokens: int
) -> list[float]:
    """
    Compute Jaccard similarity of top-k_tokens tokens between *all* pairs of
    non-dead features across two SAEs (cross-product), skipping pairs where
    either side has no tokens.
    """
    tpf_a = stats_a.get("tokens_per_feature")
    tpf_b = stats_b.get("tokens_per_feature")
    act_a = stats_a.get("activation_counts")
    act_b = stats_b.get("activation_counts")
    if tpf_a is None or tpf_b is None or act_a is None or act_b is None:
        return []

    live_a = [
        (i, _topk_token_set(tpf_a[i], k_tokens))
        for i in range(len(act_a))
        if act_a[i].item() > 0
    ]
    live_b = [
        (j, _topk_token_set(tpf_b[j], k_tokens))
        for j in range(len(act_b))
        if act_b[j].item() > 0
    ]

    scores: list[float] = []
    # Early exit if no live features
    if not live_a or not live_b:
        return scores

    # Cross-product; sets are small (<= k_tokens), so this is cheap.
    for _, set_a in live_a:
        if not set_a:
            continue
        len_a = len(set_a)
        for _, set_b in live_b:
            if not set_b:
                continue
            inter = len(set_a & set_b)
            if inter == 0:
                # Union is len_a + len_b when disjoint.
                scores.append(0.0)
                continue
            len_b = len(set_b)
            union = len_a + len_b - inter
            if union > 0:
                scores.append(inter / union)
    return scores


def _mean_live(scores: torch.Tensor) -> float | None:
    if scores.numel() == 0:
        return None
    live = scores[~scores.isnan()]
    if live.numel() == 0:
        return None
    return float(live.mean().item())


def _topk_mean(scores: torch.Tensor, k: int) -> tuple[float | None, int]:
    """
    Mean of the top-k highest non-NaN scores. Returns (mean, used_count).
    """
    if scores.numel() == 0 or k <= 0:
        return None, 0
    live = scores[~scores.isnan()]
    if live.numel() == 0:
        return None, 0
    k = min(k, live.numel())
    topk_vals = torch.topk(live, k=k).values
    return float(topk_vals.mean().item()), int(k)


def get_level_sizes(name: str):
    """
    For q_sae / rq_sae, return list of hidden sizes per level.
    Returns None for other models or if loading fails.
    """
    if name not in {"q_sae", "rq_sae"}:
        return None

    try:
        sae = load_sae(name, device="cpu")
    except Exception as exc:  # pragma: no cover - defensive
        print(f"Warning: could not load SAE '{name}' to infer level sizes: {exc}")
        return None

    model = sae.model
    if isinstance(model, QuantizedMatryoshkaSAE):
        return list(model.decoder.nested_dictionary_size)
    if isinstance(model, ResidualQuantizedSAE):
        return list(model.sae_hidden_dims)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize dynamic SAE statistics from saved .pt files."
    )
    parser.add_argument(
        "--stats-dir",
        type=str,
        default="outputs",
        help="Directory containing dynamic_stats_*.pt files (default: outputs)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=1,
        help="Activation-count threshold for 'fraction below threshold' metric.",
    )
    args = parser.parse_args()

    root = Path(args.stats_dir)
    sae_names = ["b_sae", "q_sae", "rq_sae", "baseline_sae"]
    stats_cache = {}

    print(f"Using stats directory: {root.resolve()}")
    print(f"Activation-count threshold: {args.threshold}\n")

    # Set to True if you want per-SAE summaries printed.
    print_summaries = False

    for name in sae_names:
        stats = load_stats(root, name)
        stats_cache[name] = stats
        activation_counts = stats["activation_counts"]
        coactivation = stats["coactivation"]
        tokens_per_feature = stats.get("tokens_per_feature")
        level_sizes = get_level_sizes(name)

        mean_act = summarize_activation_counts(activation_counts)
        num_below = count_below_threshold(activation_counts, threshold=args.threshold)
        avg_cofeat = average_coactivating_features(coactivation, activation_counts)
        avg_unique_tokens = average_unique_tokens_per_active_feature(
            tokens_per_feature, activation_counts
        )

        if print_summaries:
            print(f"=== Summary for {name} ===")
            print(f"Average activation count per feature: {mean_act:.4f}")
            print(
                f"Number of features with activation count < {args.threshold}: "
                f"{num_below}"
            )
            print(
                "Average number of distinct co-activating features per feature: "
                f"{avg_cofeat:.4f}"
            )
            if tokens_per_feature is not None:
                print(
                    "Average unique tokens per active feature: "
                    f"{avg_unique_tokens:.4f}"
                )
            else:
                print("Average unique tokens per active feature: [tokens not provided]")
            if level_sizes:
                print("Per-level breakdown:")
                start = 0
                for level_idx, size in enumerate(level_sizes):
                    sl = slice(start, start + size)
                    act_slice = activation_counts[sl]
                    # For co-activation, keep rows for this level but include
                    # co-activations with all features (so cross-level co-acts are counted).
                    level_row_mask = torch.zeros_like(
                        activation_counts, dtype=torch.bool
                    )
                    level_row_mask[sl] = True
                    tokens_slice = (
                        tokens_per_feature[start : start + size]
                        if tokens_per_feature is not None
                        else None
                    )

                    mean_act_lvl = summarize_activation_counts(act_slice)
                    num_below_lvl = count_below_threshold(
                        act_slice, threshold=args.threshold
                    )
                    avg_cofeat_lvl = average_coactivating_features(
                        coactivation, activation_counts, row_mask=level_row_mask
                    )
                    avg_unique_tokens_lvl = average_unique_tokens_per_active_feature(
                        tokens_slice, act_slice
                    )

                    print(f"  Level {level_idx}:")
                    print(
                        f"    Average activation count per feature: "
                        f"{mean_act_lvl:.4f}"
                    )
                    print(
                        f"    Features with activation count < {args.threshold}: "
                        f"{num_below_lvl}"
                    )
                    print(
                        "    Average number of distinct co-activating features "
                        f"per feature: {avg_cofeat_lvl:.4f}"
                    )
                    if tokens_slice is not None:
                        print(
                            "    Average unique tokens per active feature: "
                            f"{avg_unique_tokens_lvl:.4f}"
                        )
                    else:
                        print(
                            "    Average unique tokens per active feature: "
                            "[tokens not provided]"
                        )
                    start += size
            print()

    # Pairwise Jaccard across different SAEs (print only)
    available = [n for n in sae_names if n in stats_cache]
    if len(available) >= 2:
        print(
            "=== Pairwise Jaccard (cross-product over non-dead features; "
            "token top-k=10 inside Jaccard) ==="
        )
        for a_name, b_name in combinations(available, 2):
            stats_a = stats_cache[a_name]
            stats_b = stats_cache[b_name]
            # Compute Jaccard with token top-k = 10
            raw_scores = jaccard_between_saes(stats_a, stats_b, k_tokens=100)
            if not raw_scores:
                print(
                    f"[Jaccard] {a_name} vs {b_name}: no comparable features."
                )
                continue

            scores_tensor = torch.tensor(raw_scores, dtype=torch.float32)

            mean_all = float(scores_tensor.mean().item()) if scores_tensor.numel() > 0 else None
            top10_vals = nlargest(10, raw_scores)
            top100_vals = nlargest(100, raw_scores)
            top1000_vals = nlargest(1000, raw_scores)
            top10000_vals = nlargest(10000, raw_scores)

            mean_top10 = float(sum(top10_vals) / len(top10_vals)) if top10_vals else None
            mean_top100 = float(sum(top100_vals) / len(top100_vals)) if top100_vals else None
            mean_top1000 = float(sum(top1000_vals) / len(top1000_vals)) if top1000_vals else None
            mean_top10000 = float(sum(top10000_vals) / len(top10000_vals)) if top10000_vals else None
            used10 = len(top10_vals)
            used100 = len(top100_vals)
            used1000 = len(top1000_vals)
            used10000 = len(top10000_vals)

            print(f"[Jaccard] {a_name} vs {b_name}:")
            print(
                f"  mean (all live pairs): {mean_all:.4f} over {scores_tensor.numel()} pairs"
            )
            print(
                f"  mean of top-10 scores:  {mean_top10:.4f} over {used10} pairs"
                if mean_top10 is not None
                else "  mean of top-10 scores:  no live pairs"
            )
            print(
                f"  mean of top-100 scores: {mean_top100:.4f} over {used100} pairs"
                if mean_top100 is not None
                else "  mean of top-100 scores: no live pairs"
            )
            print(
                f"  mean of top-1000 scores: {mean_top1000:.4f} over {used1000} pairs"
                if mean_top1000 is not None
                else "  mean of top-1000 scores: no live pairs"
            )
            print(
                f"  mean of top-10000 scores: {mean_top10000:.4f} over {used10000} pairs"
                if mean_top10000 is not None
                else "  mean of top-10000 scores: no live pairs"
            )


if __name__ == "__main__":
    main()


