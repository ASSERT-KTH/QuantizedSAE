import torch
from torch.utils.data import Dataset, DataLoader

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