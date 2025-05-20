import torch
import os
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import matplotlib.pyplot as plt

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
        self.shift_factor = (2**(self.n_bits - 1))
        self.scale_factor = self.shift_factor / (self.gamma + 1e-5)
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
        return self.quantize(sample)
        # return self.quantize_weighted(sample)
    
    def quantize(self, sample):

        scaled_sample = sample * self.scale_factor + self.shift_factor 
        scaled_sample = torch.clamp(scaled_sample, 0, 2**self.n_bits - 1).round().int()

        bit_positions = torch.arange(0, self.n_bits, device=scaled_sample.device)
    
        scaled_sample_expanded = scaled_sample.unsqueeze(-1)
    
        binary_repr = ((scaled_sample_expanded >> bit_positions) & 1).float()
        # binary_repr = binary_repr * 2 - 1
    
        return binary_repr.view(-1)

    def quantize_weighted(self, sample):

        scaled_sample = sample * self.scale_factor + self.shift_factor 
        scaled_sample = torch.clamp(scaled_sample, 0, 2**self.n_bits - 1).round().int()

        bit_positions = torch.arange(0, self.n_bits, device=scaled_sample.device)
    
        scaled_sample_expanded = scaled_sample.unsqueeze(-1)
    
        # binary_repr = ((scaled_sample_expanded >> bit_positions) & 1).float()
        binary_repr = (scaled_sample_expanded & (1 << bit_positions)).float() / (1 << bit_positions).sum()
    
        return binary_repr.view(-1)

def compute_statistic(dataset, batch_size=1000):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    min_val, max_val = None, None
    n_samples = 0
    mean = 0.0
    M2 = 0.0  # For variance calculation
    data_array = torch.tensor([])

    for batch in loader:
        # Assume batch is a tuple (features, target), adjust as needed
        if isinstance(batch, (list, tuple)):
            data = batch[0]
        else:
            data = batch

        batch = data.view(-1)  # Flatten if needed
        num_items = batch.numel()
        k = max(1, num_items // 1000)
        indices = torch.randperm(num_items)[:k]
        data_array = torch.concat((data_array, batch[indices]))

        # batch_min = batch.min()
        # batch_max = batch.max()

        # if min_val is None:
        #     min_val = batch_min
        #     max_val = batch_max
        # else:
        #     min_val = torch.min(min_val, batch_min)
        #     max_val = torch.max(max_val, batch_max)

        # batch_samples = batch.size(0)
        # n_samples += batch_samples

        # batch_mean = batch.mean()
        # batch_var = batch.var(unbiased=False)

        # delta = batch_mean - mean
        # mean += delta * batch_samples / n_samples
        # M2 += batch_var * batch_samples + (delta ** 2) * batch_samples * (n_samples - batch_samples) / n_samples

    plt.figure(figsize=(10, 6))
    plt.hist(data_array, bins=256, density=True, alpha=0.7)
    plt.title('Downsampled Plot')
    plt.savefig("Histogram of activations")
    # shapiro_test = stats.shapiro(data_array)
    # print(f"Shapiro-Wilk Test: statistic={shapiro_test[0]:.4f}, p-value={shapiro_test[1]:.4f}")
    # print(f"Interpretation: {'Data appears normally distributed' if shapiro_test[1] > 0.05 else 'Data does not appear normally distributed'}")

    # std = torch.sqrt(M2 / n_samples)
    # return mean, std, max_val, min_val

# hidden_state_dataset = HiddenStatesTorchDatasetInBinary(os.path.join("dataset/", "the_pile_hidden_states_L3_35.pt"), 4, 4)
# print(hidden_state_dataset.__getitem__(102).shape)
# print(hidden_state_dataset.__getitem__(102))
# print(compute_statistic(hidden_state_dataset))