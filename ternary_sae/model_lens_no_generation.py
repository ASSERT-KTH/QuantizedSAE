from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch

# device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

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
inputs_tokens = tokenizer.batch_decode(inputs["input_ids"].view(4, 1))

print("Input text:", inputs_tokens)
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

# After running the forward pass (outputs = model(...))
hidden_states = outputs.hidden_states
attentions = outputs.attentions

for i, hs in enumerate(hidden_states):
    logits = model.embed_out(hs[0])
    predicted_token_ids = torch.argmax(logits, dim=-1)
    predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
    print("Predicted text:", predicted_tokens)

print("\nAttention Weights:")
for i, attn in enumerate(attentions):
    print(f"Layer {i+1}: {attn.shape}")  # Shape: (batch_size, num_heads, seq_length, seq_length)

outputs = model.generate(**inputs, output_hidden_states=True, output_attentions=True)