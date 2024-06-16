'''

Calculate simple evaluation, compare two lists.


'''

import re
from transformers import BertModel, BertTokenizer
import torch
import torch.nn.functional as F


# Function to calculate cosine similarity
def cosine_similarity(embedding1, embedding2):
    return F.cosine_similarity(embedding1, embedding2, dim=0).item()

# Function to encode phrases using BERT
def encode_phrases(phrases, model, tokenizer):
    encoded = []
    for phrase in phrases:
        inputs = tokenizer(phrase, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)
        # Use the mean of the last hidden states as the phrase embedding
        encoded.append(outputs.last_hidden_state.mean(dim=1).squeeze())
    return encoded

# Main function to match embeddings
def match_embedding(pred_list, gold_list):
    # Load pretrained model and tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')

    # Encode phrases
    pred_encoded = encode_phrases(pred_list, model, tokenizer)
    gold_encoded = encode_phrases(gold_list, model, tokenizer)

    # Calculate cosine similarity for each pair
    total_similarity = 0
    count = 0
    for pred_emb in pred_encoded:
        for gold_emb in gold_encoded:
            similarity = (cosine_similarity(pred_emb, gold_emb) + 1) / 2  # Normalizing to [0, 1]
            total_similarity += similarity
            count += 1

    # Calculate mean similarity
    mean_similarity = total_similarity / count if count else 0
    return mean_similarity

def extract_pred_topics(line):
    # Regular expression to match the pattern "number. concept"
    # This regex looks for a digit followed by a period or a digit, then captures any text until the next digit or end of line
    matches = re.findall(r'\d+\.\s*([^0-9]+)', line)

    # Process matches to clean and extract the desired concepts
    extracted_concepts = [match.strip().lower() for match in matches]

    final_extracted_concepts = []
    for match in extracted_concepts:
        item = match.strip().lower()
        if len(item) > 50 and ':' in item:
            item = item.split(':')[0]
        if '\t' in item:
            item = item.split('\t')[0]
        final_extracted_concepts.append(item)
    # print (final_extracted_concepts)
    return final_extracted_concepts

def extract_gold_topics(line):
    extracted_topics = line.strip().split(';')
    final_extracted_topics = []
    for topic in extracted_topics:
        item = topic.strip().lower()
        if '\t' in item:
            item = item.split('\t')[0]
        final_extracted_topics.append(item)
    return [x.lower() for x in extracted_topics]


def calculate_precision_recall_f1(pred, truth):
    # Convert lists to sets for easier calculation
    pred_set = set(pred)
    truth_set = set(truth)

    # Calculate precision, recall, and F1-score
    true_positives = len(pred_set.intersection(truth_set))
    precision = true_positives / len(pred_set) if pred_set else 0
    recall = true_positives / len(truth_set) if truth_set else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) else 0
    return precision, recall, f1_score

def calculate_embedding_score(pred,truth):

    if len(pred) == 0:
        return 0.

    return match_embedding(pred,truth)

res = {'precision':[],'recall':[],'f1':[]}
emb_res = []
# eval_path = 'output_gpt4/'
# eval_path = 'output_llama/'
eval_path = 'output_langchain/'
# eval_path = 'output_gpt3/'
eval_fname = 'M1_2_one_hop.tsv'
with open(eval_path+eval_fname) as f:
    for line in f.readlines():
        if len(line) > 2:
            pred, gold = line.strip().split('*****')
            pred_topics = extract_pred_topics(pred)
            gold_topics = extract_gold_topics(gold)
            precision, recall, f1_score = calculate_precision_recall_f1(pred_topics, gold_topics)
            emb_res.append(calculate_embedding_score(pred_topics,gold_topics))
            res['precision'].append(precision)
            res['recall'].append(recall)
            res['f1'].append(f1_score)
import pdb;pdb.set_trace()
res_len = len(res['f1'])
print ('precision:',sum(res['precision'])/res_len)
print ('recall:',sum(res['recall'])/res_len)
print ('f1',sum(res['f1'])/res_len)
print ('Emb ',sum(emb_res)/res_len)


'''
one-hop anc

llama:
precision: 0.061523809523809536
recall: 0.029195231801903466
f1 0.030212175713762314
----
gpt3:
precision: 0.06366666666666668
recall: 0.009514349970757533
f1 0.015356192080223255


one-hop pds
precision: 0.01
recall: 0.020499999999999997
f1 0.013132478632478633


multi-hop
precision: 0.02
recall: 0.04269047619047619
f1 0.024344322344322347


---one hop---
Embedding:
GPT4: 0.8132336083474678
GPT3: 0.806460965967758
llama: 0.8100448631810466
Langchain: 0.5710122833864588 [35 valid results: 0.8157318334092268]
'''

