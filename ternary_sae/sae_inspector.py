import torch
import os
from ternary_sae import TernarySparseAutoencoder

class TernarySparseAutoencoderInspector():

    def __init__(self, config):
        self.model = TernarySparseAutoencoder(**config)
        model_path = f"SAEs/t_sae_hidden_{config['hidden_dim']}.pth"
        if os.path.exists(model_path):
            print(f"{model_path} exists.")
            self.model.load_state_dict(torch.load(model_path))
    
    def printDictionary(self, first_row=10):
        weight = self.model.decoder.weight
        threshold = self.model.decoder.threshold

        sign_weight = torch.sign(weight)
        mask = (torch.abs(weight) >= threshold).float()
        hard_weights = sign_weight * mask # Ternary SAE
        dictionary_in_fp = weight.permute(1, 0).contiguous()
        dictionary_in_ternary = hard_weights.permute(1, 0).contiguous()

        for i in range(first_row):
            print(dictionary_in_fp[i])
            print(dictionary_in_ternary[i])
            pass

model_config = {
    'input_dim': 512,
    'hidden_dim': 4096
}

inspector = TernarySparseAutoencoderInspector(model_config)
inspector.printDictionary()