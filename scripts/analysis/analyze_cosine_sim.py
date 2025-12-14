from analysis.sae_analysis import (
    SAEModelSpec, load_sae_weights,
    directional_overlap, directional_overlap_within, cosine_similarity_matrix
)

baseline = load_sae_weights(SAEModelSpec(
    "baseline",
    "baseline_SAE/EleutherAI/sae-pythia-70m-32k/layers.3/sae.safetensors",
    kind="baseline",
))

binary = load_sae_weights(SAEModelSpec(
    "binary",
    "Trained_SAEs/b_sae_32768_4_bits.pth",
    kind="binary",
    binary_gamma=1.5,
))

quantized = load_sae_weights(SAEModelSpec(
    "quantized",
    "Trained_SAEs/q_sae_32768_4_bits.pth",
    kind="quantized",
    n_bits=4,
    abs_range=1.5,
))

threshold = 0.5

binary_stats, binary_extra = directional_overlap(
    binary,
    baseline,
    threshold=threshold,
    return_max_vector=True,
)

quant_stats, quant_extra = directional_overlap(
    quantized,
    baseline,
    threshold=threshold,
    return_max_vector=True,
)

print("Binary vs Baseline stats:", binary_stats)
print("Binary max similarities mean:", binary_extra["a_to_b_max"].mean())

print("Quantized vs Baseline stats:", quant_stats)
print("Quantized max similarities mean:", quant_extra["a_to_b_max"].mean())