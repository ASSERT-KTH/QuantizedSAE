import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformer_inspector import TransformerInspector

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

class MidLayerExtractor:
    def __init__(self, inspector, layer_k, device='cpu', batch_size=10, chunk_fraction=20):
        self.inspector = inspector
        self.layer_k = layer_k
        self.device = device
        self.batch_size = batch_size
        self.chunk_fraction = chunk_fraction

    def process_batch(self, output_path):
        all_hidden_states = []
        for i in range(self.chunk_fraction):
            all_hidden_states.append(torch.load(os.path.join("tmp/", f"batch_{i+1}.pt")))
        
        chunk_tensor = torch.cat(all_hidden_states, dim=0)
        torch.save(chunk_tensor, output_path)

    def process_chunk(self, chunk_path, output_path):

        chunk = torch.load(chunk_path)  # List of tensors, e.g., [tensor(250,), ...]
        token_ids = torch.stack(chunk, dim=0)  # Shape: (num_sequences, 250)

        num_sequences = token_ids.shape[0]
        progress = 0
        total = int(num_sequences / self.chunk_fraction)

        dataset = TensorDataset(token_ids)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        collected_hidden_states = []
        for batch in dataloader:
            token_ids_batch = batch[0].to(self.device)
            attention_mask_batch = torch.ones_like(token_ids_batch).to(self.device)
            input_dict = {
                "input_ids": token_ids_batch,
                "attention_mask": attention_mask_batch
            }
            self.inspector.forward_pass(inputs=input_dict, k=self.layer_k)
            hidden_state_k = self.inspector.hidden_states[-1][0].cpu()
            collected_hidden_states.append(hidden_state_k)
            self.inspector.reset_outputs()
            progress += self.batch_size
            print_progress_bar(progress % total, total, prefix=f"Batch_{int(progress / total)}: ")

            if progress % total == 0:
                concatenated_hidden_states = torch.cat(collected_hidden_states, dim=0)
                torch.save(concatenated_hidden_states, os.path.join("tmp/", f"batch_{int(progress / total)}.pt"))
                collected_hidden_states = []
            
        self.inspector.remove_hooks()
        del chunk
        del dataset
        del dataloader

        self.process_batch(output_path)

    def process_directory(self, data_dir, output_dir):

        # DEBUG:
        # k = 2

        os.makedirs(output_dir, exist_ok=True)
        chunk_files = [f for f in os.listdir(data_dir) if f.startswith('the_pile_deduplicated_4m_') and f.endswith('.pt')]
        # chunk_files = [f for f in chunk_files if int(f.replace('the_pile_deduplicated_4m_', '').replace('.pt', '')) <= k]

        for chunk_file in chunk_files:
            chunk_path = os.path.join(data_dir, chunk_file)
            output_file = chunk_file.replace('the_pile_deduplicated_4m_', f'the_pile_hidden_states_L{self.layer_k}_')
            output_path = os.path.join(output_dir, output_file)
            self.process_chunk(chunk_path, output_path)

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    inspector = TransformerInspector(
        model_name="EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./model/pythia-70m-deduped/step3000",
        device=device
    )

    layer_extractor = MidLayerExtractor(inspector, layer_k=3)
    layer_extractor.process_directory('dataset/', 'dataset/')