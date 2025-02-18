from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./model/pythia-70m-deduped/step3000",
).to(device)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./model/pythia-70m-deduped/step3000",
)

Text = "Hello, I am"

model.config.output_hidden_states = True
model.config.output_attentions = True

inputs = tokenizer(Text, return_tensors="pt").to(device)
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

print(outputs.attentions)