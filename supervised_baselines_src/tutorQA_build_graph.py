from transformers import BertModel, BertTokenizer
import torch
import random
from sklearn.cluster import KMeans
import numpy as np
import csv
from collections import Counter
from openai import OpenAI


# ---------- Defination ---------------
data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
annotation = data_path + 'final_new_annotation.csv'
csv_filename = '.res/new_graph_pair.csv'


model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

def sample_pairs(labels, N, final_pairs):
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append(idx)

    sampled_pairs = []
    for cluster_id, concepts in clusters.items():

        sampled_pairs_in_cluster = 0
        for _ in range(2 * N):
            pair = random.sample(concepts, 2)
            if pair not in final_pairs and pair not in sampled_pairs:
                sampled_pairs.append(pair)
                sampled_pairs_in_cluster += 1
                if sampled_pairs_in_cluster >= N:
                    break

        sampled_pairs_in_cluster = 0
        for _ in range(2 * N):
            other_clusters = [c for c_id, c in clusters.items() if c_id != cluster_id]
            concept_from_current_cluster = random.choice(concepts)
            concept_from_other_cluster = random.choice(random.choice(other_clusters))
            pair = [concept_from_current_cluster, concept_from_other_cluster]
            if pair not in final_pairs and pair not in sampled_pairs:
                sampled_pairs.append(pair)
                sampled_pairs_in_cluster += 1  # 增加该类别的采样计数
                if sampled_pairs_in_cluster >= N:
                    break

        print("this cluster: ", len(sampled_pairs))

    return sampled_pairs

def run():
    concept_data = dict()
    with open(concept_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|')
            concept_data[int(key)-1] = value.strip()
    print (len(concept_data)," concepts loaded.")

    # embeddings = {}
    # for key, text in concept_data.items():
    #     inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     target_emb = outputs.last_hidden_state[:, 0, :].view(-1)
    #     # target_emb = outputs.last_hidden_state.mean(dim=1).view(-1)
    #     embeddings[key] = target_emb.numpy()
    # print (len(embeddings)," embeddings generated.")


    # embeddings_list = list(embeddings.values())
    # embeddings = np.array(embeddings_list)
    # kmeans = KMeans(n_clusters=10, random_state=0).fit(embeddings)
    # cluster_counts = Counter(kmeans.labels_)

    # # 打印每个聚类组的成员数量
    # for cluster_id in range(10):  # 假设有10个聚类组
    #     print(f"Cluster {cluster_id}: {cluster_counts[cluster_id]} members")


    pos_pairs = []
    neg_pairs = []
    with open(annotation, 'r') as f:
        for line in f:
            edge = list(map(lambda x: int(x)-1, line.strip().split(',')))
            if edge[-1] == 0: # pos : 1-0=0
                pos_pairs.append(edge[:-1])
            elif edge[-1] == -1: #nag : 0-1=-1
                neg_pairs.append(edge[:-1])
            else:
                print("Pos/Neg Error")
    print (len(pos_pairs)," pos pairs generated.")
    print (len(neg_pairs)," neg pairs generated.")


    sampled_neg_pairs = random.sample(neg_pairs, len(pos_pairs))
    # final_pairs = pos_pairs + sampled_neg_pairs

    # cluster_sampled_pairs = sample_pairs(kmeans.labels_, 20, final_pairs)
    # print (len(cluster_sampled_pairs)," cluster pairs generated.")



    with open(csv_filename, 'w', newline='') as file:
        writer = csv.writer(file)
        for pair in pos_pairs:
            writer.writerow(pair + [1])  # 正例标记为1
        for pair in sampled_neg_pairs:
            writer.writerow(pair + [0])  # 负例标记为0
        # for pair in cluster_sampled_pairs:
        #     if pair not in pos_pairs and pair not in sampled_neg_pairs:
        #         writer.writerow(pair + [0])  # 聚类中采样的对视为负例

    print(f'Pairs have been written to {csv_filename}')


if __name__ == "__main__":
    run()
    print ('Done..')
