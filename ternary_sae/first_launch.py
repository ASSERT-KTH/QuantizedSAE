from transformers import GPTNeoXForCausalLM, AutoTokenizer

model = GPTNeoXForCausalLM.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./model/pythia-70m-deduped/step3000",
)

tokenizer = AutoTokenizer.from_pretrained(
  "EleutherAI/pythia-70m-deduped",
  revision="step3000",
  cache_dir="./model/pythia-70m-deduped/step3000",
)

SAE_layer = 3

print("Test")
inputs = tokenizer("Hello, I am", return_tensors="pt")
tokens = model.generate(**inputs)
print(tokens.tolist())
print(type(tokenizer.decode(tokens[0])))
