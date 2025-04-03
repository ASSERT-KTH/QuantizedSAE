import numpy as np
import torch
import os
import json
from collections import Counter
from ternary_sae import *
from detokenizer import *
from anthropic_handler import *
from sklearn.neighbors import NearestNeighbors
from kmeans_pytorch import kmeans

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
        self.dictionary_in_fp = weight.permute(1, 0).contiguous().detach()
        self.dictionary_in_ternary = hard_weights.permute(1, 0).contiguous().detach()

        print("Inspector initialized!")
    
    def get_feature(self, feature_idx):

        return self.dictionary_in_ternary[feature_idx]

    def calculate_k_nearest_features_cluster(self, k, type="cosine"):

        norms = torch.linalg.norm(self.dictionary_in_ternary, dim=1, keepdim=True)
        zero_vector_mask = (norms == 0)
        safe_norms = torch.where(zero_vector_mask, torch.ones_like(norms), norms)

        feature_normalized = self.dictionary_in_ternary / safe_norms

        if type == "cosine":
            cosine_similarity = torch.mm(feature_normalized, feature_normalized.t())
            distances = (1 - cosine_similarity).numpy()
            distances = np.maximum(distances, 0)
        elif type == "euclidean":
            distances = torch.cdist(feature_normalized, feature_normalized).numpy()
        
        knn = NearestNeighbors(n_neighbors=k, metric='precomputed')
        knn.fit(distances)

        _, indices = knn.kneighbors(distances)

        return distances, indices

    def distance(self, f1, f2, type="cosine"):

        f1_norms = torch.linalg.norm(self.dictionary_in_ternary[f1], dim=0, keepdim=True)
        f1_zero_vector_mask = (f1_norms == 0)
        f1_safe_norms = torch.where(f1_zero_vector_mask, torch.ones_like(f1_norms), f1_norms)

        f1_normalized = self.dictionary_in_ternary[f1] / f1_safe_norms

        f2_norms = torch.linalg.norm(self.dictionary_in_ternary[f2], dim=0, keepdim=True)
        f2_zero_vector_mask = (f2_norms == 0)
        f2_safe_norms = torch.where(f2_zero_vector_mask, torch.ones_like(f2_norms), f2_norms)

        f2_normalized = self.dictionary_in_ternary[f2] / f2_safe_norms

        if type == "cosine":
            distance = 1 - f1_normalized@f2_normalized
        elif type == "euclidean":
            distance = torch.sqrt(torch.sum((f1_normalized - f2_normalized)**2)) 
        
        return distance

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

    def check_same_entries(self, indices):

        if len(indices) < 2:
            return []

        matching = self.dictionary_in_ternary[indices[0]] == self.dictionary_in_ternary[indices[1]]

        for indice in indices[2:]:
            matching = (self.dictionary_in_ternary[indice] == self.dictionary_in_ternary[indices[0]]) & matching

        count = torch.sum(matching).item()
        pos = torch.nonzero(matching, as_tuple=True)
        return count, pos

    def k_means_analysis(self, num_clusters, type="cosine"):

        cluster_ids_x, cluster_centers = kmeans(
            X=self.dictionary_in_ternary, num_clusters=num_clusters, distance=type, device=torch.device('cpu')
        )

        cluster_ids_by_group = [[] for _ in range(num_clusters)]

        for i in range(len(cluster_ids_x)):
            cluster_ids_by_group[cluster_ids_x[i]].append(i)

        center_features = []
        for i in range(num_clusters):
            min_distance = 99999
            min_idx = -1

            for j in range(len(cluster_ids_by_group[i])):
                if type=="cosine":
                    distance = 1 - self.dictionary_in_ternary[cluster_ids_by_group[i][j]]@cluster_centers[i]
                elif type=="euclidean":
                    distance = torch.norm(self.dictioanry_in_ternary[cluster_ids_by_group[i][j]] - cluster_centers[i])
                
                if distance < min_distance:
                    min_distance = distance
                    min_idx = cluster_ids_by_group[i][j]
            
            center_features.append(min_idx)

        return cluster_ids_x, cluster_centers, cluster_ids_by_group, center_features

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

    def feature_labeling(self, feature_dict, dataset, features_list):

        feature_labels = {}

        for feature in features_list:
            if feature not in feature_dict:
                print(f"feature {feature} is not in the dictionary.")
                continue

            description = ""
            max_description = 20
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
    
    def sparsity_rate(self):
        dictionary_size = self.dictionary_in_ternary.numel()
        zeros = torch.sum(self.dictionary_in_ternary == 0).item()

        return zeros / dictionary_size

model_config = {
    'input_dim': 512,
    'hidden_dim': 4096
}

# chunk_files = [f for f in os.listdir("dataset/") if f.startswith('the_pile_hidden_states_L3_') and f.endswith('.pt') and int(f[len('the_pile_hidden_states_L3_'):-len('.pt')]) == 27]

inspector = TernarySparseAutoencoderInspector(model_config)

print(inspector.sparsity_rate())

# num_cluster = 16
# cluster_ids_x, cluster_centroids, cluster_ids_by_group, center_features = inspector.k_means_analysis(num_cluster)
# print(cluster_ids_x)
# print(cluster_centroids)
# print(f"The center of the {num_cluster} groups are:")
# print(center_features)
# print(cluster_ids_x[807])
# print(cluster_ids_x[3624])
# 
# batch_count = 27
# feature_activations = torch.load(f"the_pile_hidden_states_L3_{batch_count}_overview.pt")
# if os.path.exists(f"the_pile_hidden_states_L3_{batch_count}_overview_feature_dict"):
#     with open(f'the_pile_hidden_states_L3_{batch_count}_overview_feature_dict', 'wb') as f:
#         feature_dict = json.load(f)
# else:
#     feature_dict = inspector.print_feature_activations_overview(feature_activations)
    # with open(f'the_pile_hidden_states_L3_{batch_count}_overview_feature_dict', 'wb') as f:
    #     json.dump(feature_dict, f)
    # torch.save(f"the_pile_hidden_states_L3_{batch_count}_overview_feature_dict.pt")
# feature_dict = inspector.print_feature_activations_overview(feature_activations)
# detokenizer = TokenDetokenizer()
# token_id_lst = detokenizer.load_dataset(f"dataset/the_pile_deduplicated_4m_{batch_count}.pt")
# token_lst = detokenizer.detokenize_batch(token_id_lst)
# feature_labels = inspector.feature_labeling(feature_dict, token_lst, center_features)
# inspector.save_features_json(feature_labels, f"feature_labels/interpretation_of_{num_cluster}_clusters_centers.json")

# Feature analysis in general
# inspector.get_feature(807)
# k = 10
# distances, indices = inspector.calculate_k_nearest_features_cluster(10)
# print(indices[3624])
# print(indices[807])
# 
# distances = []
# cluster_result = cluster_ids_x[indices[807]]
# for indice in indices[807, 1:]:
#     distances.append(inspector.distance(807, indice).item())
# 
# count, pos = inspector.check_same_entries(indices[807])
# 
# print(f"For feature 807, its {k} nearest neighbors are clustered as:")
# print(cluster_result)
# print(f"There are {count} entries keeping the same for feature 807's {k} nearest neighbors.")
# print(pos)

# for f in chunk_files:
#     data = torch.load(os.path.join("dataset/", f), map_location='cpu')
#     num_contexts, tokens_per_context, feature_dim = data.shape
#     feature_activations = inspector.linguistic_analyze(data)
#     torch.save(feature_activations, f[:-3]+"_overview.pt")
#     break
# 
# batch_count = 27
# feature_activations = torch.load("the_pile_hidden_states_L3_{batch_count}_overview.pt")
# if os.path.exists(f"the_pile_hidden_states_L3_{batch_count}_overview_feature_dict.pt"):
#     feature_dict = torch.load(f"the_pile_hidden_states_L3_{batch_count}_overview_feature_dict.pt")
# else:
#     feature_dict = inspector.print_feature_activations_overview(feature_activations)
#     torch.save("the_pile_hidden_states_L3_{batch_count}_overview_feature_dict.pt")
# 
# print(f"Total amount of most activated features is: {len(torch.unique(torch.tensor(feature_activations)))}.")
# detokenizer = TokenDetokenizer()
# token_id_lst = detokenizer.load_dataset(f"dataset/the_pile_deduplicated_4m_{batch_count}.pt")
# token_lst = detokenizer.detokenize_batch(token_id_lst)
# feature_labels = inspector.feature_labeling(feature_dict, token_lst, indices[807, 1:])
# inspector.save_features_json(feature_labels, f"feature_labels/top_{k}_nearest_features_of_feature_807.json")

# inspector.check_specificity(feature_dict, token_lst, ["it", "It"], 3624)
# inspector.check_sensitivity(feature_activations, token_lst, ["it", "It"], 3624)

# inspector.check_specificity(feature_dict, token_lst, ["I", "Me", "me"], 807)
# inspector.check_sensitivity(feature_activations, token_lst, ["I", "Me", "me"], 807)
# 
# inspector.check_specificity(feature_dict, token_lst, ["I", "Me", "me", "We", "we", "us"], 807)
# inspector.check_sensitivity(feature_activations, token_lst, ["I", "Me", "me", "We", "we", "us"], 807)
# print("Begin labeling......")
# feature_labels = inspector.feature_labeling(feature_dict, token_lst)
# inspector.save_features_json(feature_labels, f"feature_labels/batch_{batch_count}_sae_{model_config['hidden_dim']}.json")
# 
# inspector.print_dictionary()
# inspector.analyze_ternary_distribution()
# zero_entries = inspector.zero_entries()
# duplicates = inspector.count_duplicates()