from transformers import AutoTokenizer
import torch

class TokenDetokenizer:
    def __init__(self, model_name="EleutherAI/pythia-70m-deduped", revision="step3000", cache_dir="./model/pythia-70m-deduped/step3000"):
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir
        )
    
    def detokenize_batch(self, token_id_batches, skip_special_tokens=True):
        return list(map(lambda ids: self.tokenizer.batch_decode(ids, skip_special_tokens=skip_special_tokens), token_id_batches))
    
    def detokenize_single(self, token_ids, skip_special_tokens=True):
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)
    
    def load_dataset(self, file_path):
        import torch
        return torch.load(file_path)

# batch_count = 35
# token_id_lst = torch.load(f"dataset/the_pile_deduplicated_4m_{batch_count}.pt")
# 
# token_lst = list(map(decode_with_skip, token_id_lst))
# print(token_lst[10])
# print(tokenizer.batch_decode(token_id_lst)[10])