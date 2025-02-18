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

Text = "Hello, I am"

print("HiHi")
model.config.output_hidden_states = True
model.config.output_attentions = True

inputs = tokenizer(Text, return_tensors="pt")
print(inputs)
outputs = model(**inputs, output_hidden_states=True, output_attentions=True)
# tokens = model.generate(**inputs)
# print(tokens.tolist())

for output in outputs:
  print(type(tokenizer.decode(output)))
