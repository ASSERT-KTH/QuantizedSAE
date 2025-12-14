import torch
from torch.utils.data import DataLoader, Subset

from sae_inference_framework import load_sae
from hidden_state_dataset import HiddenStatesTorchDataset
from sae_dynamic_analysis import analyze_dataset

if __name__ == "__main__":
    hs_path = "dataset/the_pile_hidden_states_L3_9.pt"
    tok_path = "dataset/the_pile_deduplicated_4m_9.pt"

    # Hidden states: [40000, 250, 512] → dataset over 40000*250 tokens
    hs_dataset = HiddenStatesTorchDataset(hs_path)
    _, num_contexts, tokens_per_context, _ = hs_dataset.files_info

    # Use only the first 1/10 of tokens for validation to reduce compute
    total_tokens = len(hs_dataset)
    subset_size = max(1, total_tokens // 10)
    hs_subset = Subset(hs_dataset, range(subset_size))
    loader = DataLoader(hs_subset, batch_size=32768, shuffle=False)

    # Tokens: list of length 40000, each [250]
    tok_list = torch.load(tok_path, map_location="cpu")
    token_ids = torch.stack(tok_list, dim=0)   # [40000, 250]

    for name in ["b_sae"]:
        sae = load_sae(name, device="cuda")
        print(f"\n=== Dynamic analysis for {name} ===")

        stats = analyze_dataset(
            sae,
            loader,
            token_ids=token_ids,
            tokens_per_context=tokens_per_context,
            device="cuda",
        )

        print("Reconstruction MSE (final):", stats["mse_final"])

        if stats["mse_per_level"] is not None:
            print("Per-level reconstruction MSE:", stats["mse_per_level"].tolist())
        if stats["l0_per_level"] is not None:
            print("Per-level average L0 (activations per token):", stats["l0_per_level"].tolist())

        torch.save(stats, f"outputs/dynamic_stats_{name}.pt")
        print("Total activations:", int(stats["activation_counts"].sum().item()))