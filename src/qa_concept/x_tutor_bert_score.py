# Step 1: Import the required libraries
import sys

from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

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


def retrieve_relevant(p, y, model, tokenizer, threshold=0.8):
    relevant = y
    retrieve = p
    relevant_retrieve = []
    for pp in p:
        if pp in y:
            relevant_retrieve.append(pp)
        else:
            sim_list = np.array([similarity(text1=yy, text2=pp, model=model, tokenizer=tokenizer) for yy in y])
            if any(sim_list >= threshold):
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

    print(precision, recall)

    return precision, recall


def main(threshold):
    df = pd.read_csv('./tutor_qa_compare.csv')

    refer = df['Reference Concepts'].values
    gpt_4 = df['GPT-4'].values
    gpt_35 = df['GPT-3.5'].values
    llama = df['Llama-2-70b'].values

    # Step 2: Load the pre-trained BERT model and tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased")

    save_df = []
    for r, g4, g35, la in tqdm(zip(refer, gpt_4, gpt_35, llama), total=len(df)):
        # process
        r = r.replace(';;', ';').split(';')
        g4 = g4.replace(';;', ';').split(';')
        g35 = g35.replace(';;', ';').split(';')
        la = la.replace(';;', ';').split(';')

        print('g4')
        precision_g4, recall_g4 = retrieve_relevant(p=g4, y=r, model=model, tokenizer=tokenizer, threshold=threshold)
        print('g35')
        precision_g35, recall_g35 = retrieve_relevant(p=g35, y=r, model=model, tokenizer=tokenizer, threshold=threshold)
        print('llama')
        precision_la, recall_la = retrieve_relevant(p=la, y=r, model=model, tokenizer=tokenizer, threshold=threshold)

        save_df.append([precision_g4, recall_g4, precision_g35, recall_g35, precision_la, recall_la])

    save_df = pd.DataFrame(
        save_df,
        columns=['precision_g4', 'recall_g4', 'precision_g35', 'recall_g35', 'precision_la', 'recall_la']
    )
    save_df.to_csv('./tutor_precision_recall_threshold_{}.csv'.format(threshold), index=False)


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
    # for i in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
    #     print('IN THRESHOLD', i)
    #     main(threshold=i)

    check_precision_recall()

    print ('Finished.')

