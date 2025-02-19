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

attention_outputs = []
mlp_outputs = []
hidden_states = []

n_layer = model.config.num_hidden_layers

def attention_hook(module, input, output):
    attention_outputs.append(output[0].detach().cpu())

def mlp_hook(module, input, output):
    mlp_outputs.append(output.detach().cpu())

def hidden_state_hook(module, input, output):
    hidden_states.append(output)

hooks = []
for layer in model.gpt_neox.layers:
    hooks.append(layer.attention.register_forward_hook(attention_hook))
    hooks.append(layer.mlp.register_forward_hook(mlp_hook))
    hooks.append(layer.register_forward_hook(hidden_state_hook))

Text = "Hello, I am"

inputs = tokenizer(Text, return_tensors="pt").to(device)
inputs_tokens = tokenizer.batch_decode(inputs["input_ids"].view(-1, 1))
print("Input text:", inputs_tokens)

outputs = model.generate(**inputs, max_length = 16)

for hook in hooks:
    hook.remove()

# for i, hidden_state in enumerate(hidden_states):
#     print(hidden_state[0].shape)
#     logits = model.embed_out(hidden_state[0])
#     predicted_token_ids = torch.argmax(logits, dim=-1)
#     predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
# 
#     if (i < n_layer):
#         print(f"0th to {len(inputs_tokens)}th tokens in layer {i%n_layer+1} is:", predicted_tokens)
#     else:
#         print(f"{i//n_layer+len(inputs_tokens)}th token in layer {i%n_layer+1} is:", predicted_tokens)

predictions = {}
    
for i, hidden_state in enumerate(hidden_states):
    token_position = i // n_layer
    layer_num = i % n_layer + 1
    
    if i < n_layer:
        for j in range(len(inputs_tokens)):
            pos = token_position + j
            logits = model.embed_out(hidden_state[0][0][j])
            predicted_token_ids = torch.argmax(logits, dim=-1)
            predicted_tokens = tokenizer.batch_decode([predicted_token_ids])
            if pos not in predictions:
                predictions[pos] = []
            predictions[pos].append(predicted_tokens)
    else:
        token_position += len(inputs_tokens)
        logits = model.embed_out(hidden_state[0])
        predicted_token_ids = torch.argmax(logits, dim=-1)
        predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
        if token_position not in predictions:
            predictions[token_position] = []
        predictions[token_position].append(predicted_tokens)

for i in range(n_layer):
    print(f"Layer {i}: ", end="")
    for pos in predictions:
        print(predictions[pos][i], end=" ")
    print()

# for i, mlp_out in enumerate(mlp_outputs):
#     print(mlp_out.shape)
#     logits = model.embed_out(mlp_out[0])
#     predicted_token_ids = torch.argmax(logits, dim=-1)
#     predicted_tokens = tokenizer.batch_decode(predicted_token_ids)
# 
#     if (i < len(inputs_tokens)):
#         print(f"0th to {len(inputs_tokens)}th tokens in layer {i%6+1} is:", predicted_tokens)
#     else:
#         print(f"{i//6}th token in layer {i%6+1} is:", predicted_tokens)

print(tokenizer.decode(outputs[0]))
# print("\nAttention Weights:")
# for i, attn_out in enumerate(attention_outputs):
#     print(f"Layer {i+1}: {attn_out.shape}")  # Shape: (batch_size, num_heads, seq_length, seq_length)