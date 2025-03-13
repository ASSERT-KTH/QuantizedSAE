import torch
import os
import json
from collections import Counter
from ternary_sae import *
from detokenizer import *
from anthropic_handler import *

def print_progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=50, fill='█'):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end='\r')
    if iteration == total:
        print()

class TernarySparseAutoencoderInspector():

    def __init__(self, config):
        self.sae = TernarySparseAutoencoder(**config)
        self.agent = AnthropicHandler(model="claude-3-haiku-20240307")
        model_path = f"SAEs/t_sae_hidden_{config['hidden_dim']}.pth"
        if os.path.exists(model_path):
            print(f"{model_path} exists.")
            self.sae.load_state_dict(torch.load(model_path))
        else:
            return

        weight = self.sae.decoder.weight
        threshold = self.sae.decoder.threshold

        sign_weight = torch.sign(weight)
        mask = (torch.abs(weight) >= threshold).float()
        hard_weights = sign_weight * mask # Ternary SAE
        self.dictionary_in_fp = weight.permute(1, 0).contiguous()
        self.dictionary_in_ternary = hard_weights.permute(1, 0).contiguous()

        print("Inspector initialized!")
    
    def analyze_ternary_distribution(self):

        neg_count = (self.dictionary_in_ternary == -1).sum().item()
        zero_count = (self.dictionary_in_ternary == 0).sum().item()
        pos_count = (self.dictionary_in_ternary == 1).sum().item()
        total = self.dictionary_in_ternary.numel()

        print(f"-1s' count is: {neg_count}")
        print(f"0s' count is: {zero_count}")
        print(f"1s' count is: {pos_count}")
        print(f"total's count is: {total}")
    
    def analyze_fp_distribution(self):
        pass

    def zero_entries(self):
        zero_rows = (self.dictionary_in_ternary == 0).all(dim=1)

        return zero_rows.sum().item()

    def count_duplicates(self):
        rows = self.dictionary_in_ternary.tolist()

        row_counts = Counter(map(tuple, rows))
        duplicates = 0

        for item, count in row_counts.items():
            if count > 1:
                duplicates += 1
                print(item)

        return duplicates

    def print_dictionary(self, first_row=10):

        for i in range(first_row):
            print(self.dictionary_in_fp[i])
            print(self.dictionary_in_ternary[i])
    
    def linguistic_analyze(self, dataset):
        
        feature_activations = []
        progress = 0
        with torch.no_grad():
            for context in dataset:
                h, _ = self.sae(context)
                _, indices = torch.max(h, dim=1)
                feature_activations.append(indices.tolist())
                progress += 1
                print(progress)
        
        return feature_activations
    
    def print_feature_activations_overview(self, feature_activations):
        feature_dict = {}

        print(f"There are {len(feature_activations)} lines and each line with {len(feature_activations[0])} tokens.")
        print("Start analyzing the feature activation......")
        for line, id_lst in enumerate(feature_activations):
            for pos, id in enumerate(id_lst):
                if id in feature_dict:
                    feature_dict[id]["cnt"] += 1
                    # if not any(tup[0] == line for tup in feature_dict[id]["pos"]):
                    #     feature_dict[id]["pos"].append((line, pos))
                    feature_dict[id]["pos"].append((line, pos))
                else:
                    feature_dict[id] = {"cnt": 1, "pos": [(line, pos)]}
                
                print_progress_bar(line*len(id_lst)+pos, len(id_lst)*len(feature_activations))
                
        print(f"There are {len(feature_dict)} features fire as the most activated feature for at least once.")

        # cnt_dict = {key: value['cnt'] for key, value in feature_dict.items()}
        # print(cnt_dict)
        return feature_dict
    
    def evaluate_feature(self, description, feature_idx):
        prompt = f"""
            ### Feature #{feature_idx} Analysis
            {description}

            Based on the examples above, please provide:

            1. A concise summary label describing what linguistic or semantic property this feature represents.
            2. A brief detailed explanation supporting your interpretation.

            Your response should follow this format:

            Feature #{feature_idx}:
            - **Feature summary:** [Your concise label here]
            - **Detailed explanation:** [Your brief explanation here]
            """

        return self.agent.get_response(prompt)

    def feature_labeling(self, feature_dict, dataset):

        feature_labels = {}

        max_description = 20
        max_features = 5

        feature_cnt = 0

        for feature in feature_dict:
            description = ""

            prev_line = -1
            description_cnt = 0

            for pos_inf in feature_dict[feature]['pos']:
                line = pos_inf[0]
                pos = pos_inf[1]
                if line != prev_line:
                    description += "\nOriginal sequence: " + ''.join(dataset[line])
                    prev_line = line
                # else:
                #     continue
                description += "'\nMost activated token is {" + dataset[line][pos] + f"}} in position {pos}."
                description_cnt += 1

                if description_cnt >= max_description:
                    break

            feature_labels[feature]= {"description": description, "interpretation": self.evaluate_feature(description, feature)}
            feature_cnt += 1

            if feature_cnt > max_features:
                break
        
        return feature_labels
    
    def save_features_json(self, feature_labels, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(feature_labels, f, 
                     indent=4, ensure_ascii=False,
                     default=lambda o: o.__dict__)

    def check_sensitivity(self, feature_activations, dataset, target_tokens, feature_id):
        sensitivity_score = 0
        target_token_appearance = 0

        for line, token_seq in enumerate(dataset):
            for pos, token in enumerate(token_seq):
                if any(target_token in token for target_token in target_tokens):
                    target_token_appearance += 1
                    if feature_activations[line][pos] == feature_id:
                        sensitivity_score += 1
        
        print(f"Target tokens has appeared {target_token_appearance} times in the dataset.")
        print(f"The feature {feature_id} as mostly activated {sensitivity_score}.")

        return sensitivity_score / target_token_appearance

    def check_specificity(self, feature_dict, dataset, target_tokens, feature_id):
        specificity_score = 0

        for (line, pos) in feature_dict[feature_id]["pos"]:
            if any(target_token in dataset[line][pos] for target_token in target_tokens):
                specificity_score += 1
        
        print(f"The feature {feature_id} has been the most activated feature for {feature_dict[feature_id]['cnt']} times in the dataset.")
        print(f"The feature {feature_id} activated {specificity_score} times on the target tokens.")

        return specificity_score / feature_dict[feature_id]["cnt"]

model_config = {
    'input_dim': 512,
    'hidden_dim': 4096
}

chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt') and int(f[len('the_pile_hidden_states_L3_'):-len('.pt')]) == 27]

inspector = TernarySparseAutoencoderInspector(model_config)

for f in chunk_files:
    data = torch.load(os.path.join("dataset/", f), map_location='cpu')
    num_contexts, tokens_per_context, feature_dim = data.shape
    feature_activations = inspector.linguistic_analyze(data)
    torch.save(feature_activations, f[:-3]+"_overview.pt")
    break

batch_count = 27
feature_activations = torch.load("the_pile_hidden_states_L3_27_overview.pt")
feature_dict = inspector.print_feature_activations_overview(feature_activations)

print(f"Total amount of most activated features is: {len(torch.unique(torch.tensor(feature_activations)))}.")
detokenizer = TokenDetokenizer()
token_id_lst = detokenizer.load_dataset(f"dataset/the_pile_deduplicated_4m_{batch_count}.pt")
token_lst = detokenizer.detokenize_batch(token_id_lst)

# inspector.check_specificity(feature_dict, token_lst, ["it", "It"], 3624)
# inspector.check_sensitivity(feature_activations, token_lst, ["it", "It"], 3624)

inspector.check_specificity(feature_dict, token_lst, ["I", "Me", "me"], 807)
inspector.check_sensitivity(feature_activations, token_lst, ["I", "Me", "me"], 807)

inspector.check_specificity(feature_dict, token_lst, ["I", "Me", "me", "We", "we", "us"], 807)
inspector.check_sensitivity(feature_activations, token_lst, ["I", "Me", "me", "We", "we", "us"], 807)
# print("Begin labeling......")
# feature_labels = inspector.feature_labeling(feature_dict, token_lst)
# inspector.save_features_json(feature_labels, f"feature_labels/batch_{batch_count}_sae_{model_config['hidden_dim']}.json")
# 
# inspector.print_dictionary()
# inspector.analyze_ternary_distribution()
# zero_entries = inspector.zero_entries()
# duplicates = inspector.count_duplicates()