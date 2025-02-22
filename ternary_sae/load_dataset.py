from datasets import load_dataset
from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    # Print a newline when the progress is complete
    if iteration == total:
        print()

# n_contexts = 40000000
n_contexts = 4000000 
# n_contexts = 1000 
len_sample = 250

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./model/pythia-70m-deduped/step3000",
)

dataset = load_dataset("EleutherAI/the_pile_deduplicated", split="train", streaming=True)
shuffled_dataset = dataset.shuffle(buffer_size=10000, seed=42)

# trainingset = {"input_ids": [], "attention_mask": []}
trainingset = []
batch_count = 0

for doc in dataset:
# for doc in shuffled_dataset:
    text = doc["text"]
    token_seq = tokenizer(text, return_tensors="pt")

    length = len(token_seq["input_ids"][0])

    if length < len_sample:
        continue

    start_idx = torch.randint(0, length - len_sample + 1, (1,)).item()

    # trainingset["input_ids"].append(token_seq["input_ids"][0][start_idx:start_idx+len_sample])
    # trainingset["attention_mask"].append(token_seq["attention_mask"][0][start_idx:start_idx+len_sample])
    trainingset.append(token_seq["input_ids"][0][start_idx:start_idx+len_sample])

    added_contexts = len(trainingset)
    progress = added_contexts % (n_contexts/100)

    print_progress_bar(progress, n_contexts/100, prefix='Progress:', suffix='Complete', length=50)

    if progress == 0:
        batch_count += 1
        torch.save(trainingset, f"dataset/the_pile_deduplicated_4m_{batch_count}.pt")
        trainingset = []
        print(f"Batch {batch_count} finished!")

    if batch_count == 100:
        break

# torch.save({"input_ids": trainingset["input_ids"], "attention_mask": trainingset["attention_mask"]}, "dataset/the_pile_deduplicated_1k")
# torch.save({"input_ids": trainingset["input_ids"], "attention_mask": trainingset["attention_mask"]}, "dataset/the_pile_deduplicated_4m")
# torch.save({"input_ids": trainingset["input_ids"], "attention_mask": trainingset["attention_mask"]}, "dataset/the_pile_deduplicated_40m")
print("Completed!")
