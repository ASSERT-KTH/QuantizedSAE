import torch
import os
from collections import Counter
from ternary_sae import TernarySparseAutoencoder

class TernarySparseAutoencoderInspector():

    def __init__(self, config):
        self.model = TernarySparseAutoencoder(**config)
        model_path = f"SAEs/t_sae_hidden_{config['hidden_dim']}.pth"
        if os.path.exists(model_path):
            print(f"{model_path} exists.")
            self.model.load_state_dict(torch.load(model_path))
        else:
            return

        weight = self.model.decoder.weight
        threshold = self.model.decoder.threshold

        sign_weight = torch.sign(weight)
        mask = (torch.abs(weight) >= threshold).float()
        hard_weights = sign_weight * mask # Ternary SAE
        self.dictionary_in_fp = weight.permute(1, 0).contiguous()
        self.dictionary_in_ternary = hard_weights.permute(1, 0).contiguous()
    
    def analyze_ternary_distribution(self):

        neg_count = (self.dictionary_in_ternary == -1).sum().item()
        zero_count = (self.dictionary_in_ternary == 0).sum().item()
        pos_count = (self.dictionary_in_ternary == 1).sum().item()
        total = self.dictionary_in_ternary.numel()

        print(f"-1s' count is: {neg_count}")
        print(f"0s' count is: {zero_count}")
        print(f"1s' count is: {pos_count}")
        print(f"total's count is: {total}")
    
    def analyze_fp_distribution(self):
        pass

    def zero_entries(self):
        zero_rows = (self.dictionary_in_ternary == 0).all(dim=1)

        return zero_rows.sum().item()

    def count_duplicates(self):
        rows = self.dictionary_in_ternary.tolist()

        row_counts = Counter(map(tuple, rows))
        duplicates = 0

        for item, count in row_counts.items():
            if count > 1:
                duplicates += 1
                print(item)

        return duplicates

    def print_dictionary(self, first_row=10):

        for i in range(first_row):
            print(self.dictionary_in_fp[i])
            print(self.dictionary_in_ternary[i])
            pass

model_config = {
    'input_dim': 512,
    'hidden_dim': 4096
}

inspector = TernarySparseAutoencoderInspector(model_config)
inspector.print_dictionary()
inspector.analyze_ternary_distribution()
zero_entries = inspector.zero_entries()
duplicates = inspector.count_duplicates()
print(duplicates)