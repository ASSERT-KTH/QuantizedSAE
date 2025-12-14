import torch
import os
from torch.utils.data import Dataset, DataLoader
# from scipy import stats
# import matplotlib.pyplot as plt

class HiddenStatesTorchDataset(Dataset):

    def __init__(self, file_path, transform=None):
        """
        file_paths: .pt file path. The file is expected to store a tensor 
                    with shape of (num_contexts, tokens_per_context, feature_dim) where
                    feature_dim is expected to be 512.
        transform: Optional transformation to apply to each sample.
        """
        self.data = torch.load(file_path, map_location='cpu')
        self.transform = transform
        num_contexts, tokens_per_context, feature_dim = self.data.shape
        
        self.cum_sizes = num_contexts * tokens_per_context
        self.files_info = (file_path, num_contexts, tokens_per_context, feature_dim)

    def __len__(self):
        return self.cum_sizes

    def __getitem__(self, idx):
        # Map the local index to (context index, token index)
        context_idx = idx // self.files_info[2]
        token_idx = idx % self.files_info[2] 
        sample = self.data[context_idx, token_idx, :]  # Each sample is a 512-d tensor
        
        sample = sample.float()
        return sample

class HiddenStatesTorchDatasetInBinary(Dataset):

    def __init__(self, file_path, gamma=4, n_bits=4, transform=None):
        """
        file_paths: .pt file path. The file is expected to store a tensor 
                    with shape of (num_contexts, tokens_per_context, feature_dim) where
                    feature_dim is expected to be 512.
        transform: Optional transformation to apply to each sample.
        """
        self.data = torch.load(file_path, map_location='cpu')
        self.transform = transform
        self.gamma = gamma
        self.n_bits = n_bits
        self.shift_factor = 2**(self.n_bits - 1)
        self.scale_factor = 2**(self.n_bits - 1) / (self.gamma + 1e-5)
        num_contexts, tokens_per_context, feature_dim = self.data.shape
        
        self.cum_sizes = num_contexts * tokens_per_context
        self.files_info = (file_path, num_contexts, tokens_per_context, feature_dim)

    def __len__(self):
        return self.cum_sizes

    def __getoriginalitem__(self, idx): # Map the local index to (context index, token index)
        context_idx = idx // self.files_info[2]
        token_idx = idx % self.files_info[2] 
        sample = self.data[context_idx, token_idx, :]  # Each sample is a 512-d tensor
        
        sample = sample.float()
        return sample

    def __getitem__(self, idx): # Map the local index to (context index, token index)
        context_idx = idx // self.files_info[2]
        token_idx = idx % self.files_info[2] 
        sample = self.data[context_idx, token_idx, :]  # Each sample is a 512-d tensor
        
        sample = sample.float()
        # return self.quantize(sample)
        return self.quantize_signed(sample)
        # return self.quantize_weighted(sample)
    
    def quantize(self, sample):

        scaled_sample = sample * self.scale_factor * 2 + self.shift_factor 
        scaled_sample = torch.clamp(scaled_sample, 0, 2**self.n_bits - 1).round().int()

        bit_positions = torch.arange(0, self.n_bits, device=scaled_sample.device)
    
        scaled_sample_expanded = scaled_sample.unsqueeze(-1)
        binary_repr = ((scaled_sample_expanded >> bit_positions) & 1).float()
        # binary_repr = binary_repr * 2 - 1
    
        return binary_repr.view(-1)

    def quantize_signed(self, sample):

        scaled_sample = sample * self.scale_factor
        scaled_sample = torch.clamp(scaled_sample, -2**(self.n_bits - 1), 2**(self.n_bits - 1) - 1).round().int()

        bit_positions = torch.arange(0, self.n_bits, device=scaled_sample.device)
    
        mask = (1 << self.n_bits) - 1
        scaled_sample_expanded = (scaled_sample & mask).unsqueeze(-1)
    
        binary_repr = ((scaled_sample_expanded >> bit_positions) & 1).float()
        # binary_repr = binary_repr * 2 - 1
    
        return binary_repr.view(-1)

# def compute_statistic(dataset, threshold=2, batch_size=1000):
#     """Compute and plot statistics of the dataset.
# 
#     In addition to the existing histogram plot, this function now also
#     calculates the ratio of activations whose absolute value is below the
#     provided ``threshold``.
# 
#     Args:
#         dataset (torch.utils.data.Dataset): Dataset of activations.
#         threshold (float, optional): Absolute value threshold for counting
#             activations. Defaults to ``0.1``.
#         batch_size (int, optional): Batch size for the DataLoader.
# 
#     Returns:
#         float: Ratio of activations with |value| < ``threshold`` across the
#             entire dataset.
#     """
# 
#     loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
# 
#     # Counters for the ratio calculation
#     total_activations = 0
#     activations_under_threshold = 0
#  
#     min_val, max_val = None, None
#     n_samples = 0
#     mean = 0.0
#     M2 = 0.0  # For variance calculation
#     data_array = torch.tensor([])
# 
#     for batch in loader:
#         # Assume batch is a tuple (features, target), adjust as needed
#         if isinstance(batch, (list, tuple)):
#             data = batch[0]
#         else:
#             data = batch
# 
#         batch = data.view(-1)  # Flatten if needed
# 
#         # --- Ratio calculation ------------------------------------------------
#         num_items = batch.numel()
#         total_activations += num_items
#         activations_under_threshold += (torch.abs(batch) < threshold).sum().item()
# 
#         # --- Down-sampling for the histogram ---------------------------------
#         k = max(1, num_items // 1000)
#         indices = torch.randperm(num_items)[:k]
#         data_array = torch.concat((data_array, batch[indices]))
# 
#         # batch_min = batch.min()
#         # batch_max = batch.max()
# 
#         # if min_val is None:
#         #     min_val = batch_min
#         #     max_val = batch_max
#         # else:
#         #     min_val = torch.min(min_val, batch_min)
#         #     max_val = torch.max(max_val, batch_max)
# 
#         # batch_samples = batch.size(0)
#         # n_samples += batch_samples
# 
#         # batch_mean = batch.mean()
#         # batch_var = batch.var(unbiased=False)
# 
#         # delta = batch_mean - mean
#         # mean += delta * batch_samples / n_samples
#         # M2 += batch_var * batch_samples + (delta ** 2) * batch_samples * (n_samples - batch_samples) / n_samples
# 
#     plt.figure(figsize=(10, 6))
#     plt.hist(data_array, bins=256, density=True, alpha=0.7)
#     plt.title('Downsampled Plot')
#     plt.savefig("Histogram of activations")
# 
#     # -------------------------------------------------------------------------
#     # Final ratio
#     ratio_under_threshold = (
#         activations_under_threshold / total_activations if total_activations > 0 else 0.0
#     )
# 
#     print(f"The dataset size is {total_activations}, the number of activations under threshold is {activations_under_threshold}")
#     print(
#         f"Ratio of activations with |value| < {threshold}: {ratio_under_threshold:.6f}"
#     )
# 
#     return ratio_under_threshold
# 
# # hidden_state_dataset = HiddenStatesTorchDataset(os.path.join("dataset/", "the_pile_hidden_states_L3_23.pt"))
# # print(hidden_state_dataset.__getitem__(102).shape)
# # print(hidden_state_dataset.__getitem__(102))
# # print(compute_statistic(hidden_state_dataset))