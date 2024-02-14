# Step 1: Import the required libraries
import os.path
import sys

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle


def similarity(text1, text2, model, tokenizer):
    # Step 4: Prepare the texts for BERT
    inputs1 = tokenizer(text1, return_tensors="pt", padding=True, truncation=True)
    inputs2 = tokenizer(text2, return_tensors="pt", padding=True, truncation=True)

    # Step 5: Feed the texts to the BERT model
    outputs1 = model(**inputs1)
    outputs2 = model(**inputs2)

    # Step 6: Obtain the representation vectors
    embeddings1 = outputs1.last_hidden_state.mean(dim=1).detach().numpy()
    embeddings2 = outputs2.last_hidden_state.mean(dim=1).detach().numpy()

    # Step 7: Calculate cosine similarity
    similarity = np.dot(embeddings1, embeddings2.T) / (np.linalg.norm(embeddings1) * np.linalg.norm(embeddings2))

    return similarity[0][0]


def info_retrie_precision_recall(relevant, retrieve, relevant_retrieve):
    precision = len(relevant_retrieve) * 1. / len(retrieve) # 10
    recall = len(np.unique(relevant_retrieve)) * 1. / len(relevant) # 3

    return precision, recall


def retrieve_relevant(p, y, retrieve_sim, threshold):
    relevant = y
    retrieve = p
    relevant_retrieve = []
    for pp, sim_list in zip(p, retrieve_sim):
        if pp in y:
            relevant_retrieve.append(pp)
        else:
            if any(np.array(sim_list) >= threshold):
                # suppose
                # p1 -> y1
                # p2 -> y1
                # then in relevant_retrieve, there will be two y1 in the list, and we will unique it in calculation
                the_idx = np.argmax(sim_list)
                relevant_retrieve.append(y[the_idx])
            else:
                pass

    precision, recall = info_retrie_precision_recall(
        relevant=relevant, retrieve=retrieve, relevant_retrieve=relevant_retrieve
    )

    # print(precision, recall)

    return precision, recall


def get_precision_recall(task, threshold):
    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-4', header=0)
    gpt_4 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-3.5', header=0)
    gpt_35 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='ground_truth', header=0)
    ground_truth = temp[task].apply(lambda x: x.replace('[', '').replace(']', '')).values

    with open('./g4_sim_{}.pkl'.format(task), 'rb') as f:
        g4_sim = pickle.load(f)
    with open('./g35_sim_{}.pkl'.format(task), 'rb') as f:
        g35_sim = pickle.load(f)

    pre_35, rec_35 = [], []
    pre_4, rec_4 = [], []
    for r, g4, g35, g4_s, g35_s in zip(ground_truth, gpt_4, gpt_35, g4_sim, g35_sim):
        r = r.split(';')
        g4 = g4.split(';')
        g35 = g35.split(';')

        precision, recall = retrieve_relevant(p=g35, y=r, retrieve_sim=g35_s, threshold=threshold)
        pre_35.append(precision)
        rec_35.append(recall)

        precision, recall = retrieve_relevant(p=g4, y=r, retrieve_sim=g4_s, threshold=threshold)
        pre_4.append(precision)
        rec_4.append(recall)

    f1_35 = 2 * np.mean(pre_35) * np.mean(rec_35) / (np.mean(pre_35) + np.mean(rec_35))
    f1_4 = 2 * np.mean(pre_4) * np.mean(rec_4) / (np.mean(pre_4) + np.mean(rec_4))

    print(threshold, 'gpt-35', np.mean(pre_35), np.mean(rec_35), f1_35)
    print(threshold, 'gpt-4', np.mean(pre_4), np.mean(rec_4), f1_4)


def get_precision_recall_kg(task, threshold):
    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-4', header=0)
    gpt_4 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='ground_truth', header=0)
    ground_truth = temp[task].apply(lambda x: x.replace('[', '').replace(']', '')).values

    with open('./g4_kg_sim_{}.pkl'.format(task), 'rb') as f:
        g4_sim = pickle.load(f)

    pre_35, rec_35 = [], []
    pre_4, rec_4 = [], []
    for r, g4, g4_s in zip(ground_truth, gpt_4, g4_sim):
        r = r.split(';')
        g4 = g4.split(';')

        if task == 'Task II':
            g4 = g4[:-1]

        precision, recall = retrieve_relevant(p=g4, y=r, retrieve_sim=g4_s, threshold=threshold)
        pre_4.append(precision)
        rec_4.append(recall)

    f1_4 = 2 * np.mean(pre_4) * np.mean(rec_4) / (np.mean(pre_4) + np.mean(rec_4))

    print(threshold, 'gpt-4', np.mean(pre_4), np.mean(rec_4), f1_4)


def calculate_sim(p, y, model, tokenizer):
    retrieve_sim = []
    for pp in p:
        if pp in y:
            retrieve_sim.append([1])
        else:
            sim_list = [similarity(text1=yy, text2=pp, model=model, tokenizer=tokenizer) for yy in y]
            retrieve_sim.append(sim_list)
    return retrieve_sim


def get_sim(task):
    print(task)
    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-4', header=0)
    gpt_4 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-3.5', header=0)
    gpt_35 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='ground_truth', header=0)
    ground_truth = temp[task].apply(lambda x: x.replace('[', '').replace(']', '')).values

    # Step 2: Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    g4_sim = []
    g35_sim = []
    for r, g4, g35 in tqdm(zip(ground_truth, gpt_4, gpt_35), total=len(ground_truth)):
        r = r.split(';')
        g4 = g4.split(';')
        g35 = g35.split(';')

        g4_sim.append(calculate_sim(p=g4, y=r, model=model, tokenizer=tokenizer))
        g35_sim.append(calculate_sim(p=g35, y=r, model=model, tokenizer=tokenizer))

    with open('./g4_sim_{}.pkl'.format(task), 'wb') as f:
        pickle.dump(g4_sim, f)
    with open('./g35_sim_{}.pkl'.format(task), 'wb') as f:
        pickle.dump(g35_sim, f)


def get_sim_kg(task):
    temp = pd.read_excel('./result.xlsx', sheet_name='GPT-4+KG', header=0)
    gpt_4 = temp[task].apply(lambda x: x.replace('[', '').replace(']', '').replace(', ', ';').replace('"', '')).values

    temp = pd.read_excel('./result.xlsx', sheet_name='ground_truth', header=0)
    ground_truth = temp[task].apply(lambda x: x.replace('[', '').replace(']', '')).values

    # Step 2: Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    g4_sim = []
    for r, g4 in tqdm(zip(ground_truth, gpt_4), total=len(ground_truth)):
        r = r.split(';')
        # last one is number of token, remove
        g4 = g4.split(';')

        if task == 'Task II':
            g4 = g4[:-1]

        g4_sim.append(calculate_sim(p=g4, y=r, model=model, tokenizer=tokenizer))

    with open('./g4_kg_sim_{}.pkl'.format(task), 'wb') as f:
        pickle.dump(g4_sim, f)


# Task II, Task III, Task IV
def main(task='Task II', the_list=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95]):
    print('--------')
    print(task)
    # if not os.path.exists('./g4_sim_{}.pkl'.format(task)):
    #     get_sim(task=task)
    # for threshold in the_list:
    #     get_precision_recall(task=task, threshold=threshold)

    print('  ')
    print('+kg')

    if not os.path.exists('./g4_kg_sim_{}.pkl'.format(task)):
        get_sim_kg(task=task)
    for threshold in the_list:
        get_precision_recall_kg(task=task, threshold=threshold)

    print('--------')


def check_precision_recall():
    the_list = [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    for threshold in the_list:
        print('threshold', threshold)
        f = './tutor_precision_recall_threshold_{}.csv'.format(threshold)
        the_df = pd.read_csv(f)

        print(the_df.mean(axis=0))

        print(' ')
        # break


if __name__ == "__main__":
    # for task in ['Task II', 'Task III', 'Task IV']:
    for task in ['Task II', 'Task III']:
        main(task=task, the_list=[0.4, 0.5, 0.6, 0.7, 0.8])

    # check_precision_recall()

    print ('Finished.')

# --------
# Task II
# 0.4 gpt-35 0.9966666666666667 0.6077892940392942 0.7551012237933835
# 0.4 gpt-4 0.997 0.752061272061272 0.8573800131786595
# 0.5 gpt-35 0.9631666666666666 0.5977892940392942 0.7377155233110454
# 0.5 gpt-4 0.9818333333333333 0.7456327006327007 0.8475848734620439
# 0.6 gpt-35 0.8493333333333334 0.5568397990897992 0.6726662482572442
# 0.6 gpt-4 0.8900728356610709 0.7242950382950383 0.7986721601572652
# 0.7 gpt-35 0.5884166666666666 0.4066125263625264 0.4809056640012176
# 0.7 gpt-4 0.6270017923253217 0.6221289127539128 0.6245558479476554
# 0.8 gpt-35 0.256 0.19282056832056832 0.2199634730412303
# 0.8 gpt-4 0.2365065767565768 0.3499736374736375 0.28226380002462165
# --------
# --------
# Task III
# 0.4 gpt-35 1.0 0.6970000000000001 0.8214496169711255
# 0.4 gpt-4 1.0 0.905 0.9501312335958005
# 0.5 gpt-35 0.9921666666666666 0.6970000000000001 0.8187944745929946
# 0.5 gpt-4 0.9874404761904763 0.905 0.944424558865159
# 0.6 gpt-35 0.9076666666666665 0.687 0.7820656354515051
# 0.6 gpt-4 0.8885778490515333 0.8983333333333333 0.893428961588648
# 0.7 gpt-35 0.6280555555555556 0.503 0.5586143720222014
# 0.7 gpt-4 0.6240385404069614 0.7971666666666667 0.7000575577007121
# 0.8 gpt-35 0.325 0.2503333333333333 0.2828215527230591
# 0.8 gpt-4 0.29172764078027236 0.5058333333333332 0.3700420901997017
# --------
# --------
# Task IV
# 0.4 gpt-35 1.0 0.4191111111111111 0.5906670842467899
# 0.4 gpt-4 1.0 0.560015873015873 0.7179617627008272
# 0.5 gpt-35 0.9975 0.4171111111111111 0.5882441189176452
# 0.5 gpt-4 0.998 0.558015873015873 0.7158022625957625
# 0.6 gpt-35 0.9941666666666668 0.4171111111111111 0.5876631368998413
# 0.6 gpt-4 0.9933333333333334 0.558015873015873 0.7145983185825617
# 0.7 gpt-35 0.9513333333333333 0.40652777777777777 0.5696362061793807
# 0.7 gpt-4 0.9348452380952382 0.5463095238095238 0.689617141969192
# 0.8 gpt-35 0.6988333333333334 0.33435714285714296 0.45230753099506854
# 0.8 gpt-4 0.707875 0.4667261904761905 0.5625463429836876
# --------






