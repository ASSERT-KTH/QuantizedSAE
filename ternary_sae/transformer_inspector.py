#!/usr/bin/env python3
import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

class StopForwardException(Exception):
    """Custom exception to stop the forward pass after a specific layer."""
    pass

class TransformerInspector:
    """
    A framework for loading a transformer model, registering hooks to collect
    hidden states, attention, and MLP outputs, generating text, and processing outputs.
    """

    def __init__(self, model_name, revision, cache_dir, device="cpu"):
        self.device = device
        self.model = GPTNeoXForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        ).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            cache_dir=cache_dir,
        )
        self.n_layer = self.model.config.num_hidden_layers
        
        self.attention_outputs = []
        self.mlp_outputs = []
        self.hidden_states = []
        self.hooks = []

    def reset_outputs(self):
        self.attention_outputs = []
        self.mlp_outputs = []
        self.hidden_states = []
        self.inputs = []

    def attention_hook(self, module, input, output):
        self.attention_outputs.append(output[0].detach().cpu())

    def mlp_hook(self, module, input, output):
        self.mlp_outputs.append(output.detach().cpu())

    def hidden_state_hook(self, module, input, output):
        self.hidden_states.append(output)

    def register_hooks(self, k=None):
        self.hooks = []
        if k is None:
            for layer in self.model.gpt_neox.layers:
                self.hooks.append(layer.attention.register_forward_hook(self.attention_hook))
                self.hooks.append(layer.mlp.register_forward_hook(self.mlp_hook))
                self.hooks.append(layer.register_forward_hook(self.hidden_state_hook))
        else:
            for i in range(k):
                layer = self.model.gpt_neox.layers[i]
                self.hooks.append(layer.attention.register_forward_hook(self.attention_hook))
                self.hooks.append(layer.mlp.register_forward_hook(self.mlp_hook))
                self.hooks.append(layer.register_forward_hook(self.hidden_state_hook))
            if k < self.n_layer:
                def stop_hook(module, input):
                    raise StopForwardException
                self.hooks.append(self.model.gpt_neox.layers[k].register_forward_pre_hook(stop_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate_text(self, text, max_length=16):
        self.reset_outputs()
        self.inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_length = self.inputs["input_ids"].shape[1]

        self.register_hooks()
        outputs = self.model.generate(
            **self.inputs,
            max_length=max_length,
            return_dict_in_generate=True,
            output_scores=True
        )
        self.remove_hooks()
        
        generated_sequence = outputs.sequences[0][input_length:]
        return generated_sequence
    
    def forward_pass(self, text, k=None):
        self.reset_outputs()
        self.inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        if k is None:
            self.register_hooks()
            self.model(**self.inputs, output_hidden_states=True, output_attentions=True)
            self.remove_hooks()
        else:
            assert 1 <= k <= self.n_layer, f"k must be between 1 and {self.n_layer}"
            self.register_hooks(k)
            try:
                self.model(**self.inputs)
            except StopForwardException:
                pass  # Computation stopped after layer k-1
            finally:
                self.remove_hooks()

    def predict_from_hidden_states(self):
        predictions = {}
        input_length = self.inputs["input_ids"].shape[1]
        
        for i, hidden_state in enumerate(self.hidden_states):
            token_position = i // self.n_layer
            layer_num = i % self.n_layer + 1

            if i < self.n_layer:
                for j in range(input_length):
                    pos = token_position + j
                    logits = self.model.embed_out(hidden_state[0][0][j])
                    predicted_token_id = torch.argmax(logits, dim=-1)
                    predicted_token = self.tokenizer.decode(predicted_token_id)
                    if pos not in predictions:
                        predictions[pos] = []
                    predictions[pos].append((layer_num, predicted_token))
            else:
                token_position += input_length
                logits = self.model.embed_out(hidden_state[0][0])
                predicted_token_id = torch.argmax(logits, dim=-1)
                predicted_tokens = self.tokenizer.decode(predicted_token_id)
                if token_position not in predictions:
                    predictions[token_position] = []
                predictions[token_position].append((layer_num, predicted_tokens))
        return predictions

    def display_predictions(self, predictions):
        for i in range(self.model.config.num_hidden_layers):
            if i >= len(predictions[0]):
                break
            print(f"Layer {i+1}: ", end="")
            for pos in predictions:
                print(predictions[pos][i][1], end="")
            print()
        # for pos in sorted(predictions):
        #     print(f"Token position {pos}:")
        #     for layer, token in predictions[pos]:
        #         print(f"  Layer {layer}: {token}")

if __name__ == "__main__":
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    inspector = TransformerInspector(
        model_name="EleutherAI/pythia-70m-deduped",
        revision="step3000",
        cache_dir="./model/pythia-70m-deduped/step3000",
        device=device
    )

    input_text = "Hello, I am"
    # input_text = "I want to generate a section of Python code:\ndef hello():\n"
    print("Input text:", input_text)

    # generated_sequence = inspector.generate_text(input_text, max_length=128)
    # print("Generated sequence:", inspector.tokenizer.batch_decode(outputs.sequences))

    inspector.forward_pass(input_text, k=3)

    predictions = inspector.predict_from_hidden_states()
    inspector.display_predictions(predictions)
