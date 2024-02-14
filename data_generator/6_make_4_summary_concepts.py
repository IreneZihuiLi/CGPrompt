import networkx as nx
import csv
from transformers import BertModel, BertTokenizer
import torch
from scipy.spatial.distance import cosine


data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
label_path = data_path + 'final_new_annotation.csv'


# load concept
concept_data = {}  # Initialize an empty dictionary
concept_list = []

# load concept as dict
with open(concept_path, 'r') as file:
    for line in file:
        # Split each line at the pipe character
        key, value = line.strip().split('|')
        # Convert key to an integer and strip any whitespace from the value
        concept_data[int(key)-1] = value.strip()
        concept_list.append(value.strip().lower())
# Create a reverse mapping from concept name to ID
name_to_id = {name: id for id, name in concept_data.items()}

print (len(concept_data)," concepts loaded.")




# Function to get embeddings
def get_embeddings(phrases):
    with torch.no_grad():
        return [model(**tokenizer(phrase, return_tensors='pt'))[0].mean(dim=1).squeeze() for phrase in phrases]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
full_concepts = [str(phrase) for phrase in concept_list]
full_embeddings = get_embeddings(full_concepts)
print ('Full list embedding done.')

def match_concepts(pred_concepts):
    # Load pretrained BERT model and tokenizer

    # Ensure all concepts are strings
    pred_concepts = [str(phrase) for phrase in pred_concepts]


    # Get embeddings for both lists
    pred_embeddings = get_embeddings(pred_concepts)


    # Find the most similar concepts
    matches = []
    for pred_emb in pred_embeddings:
        similarity = [1 - cosine(pred_emb, full_emb) for full_emb in full_embeddings]
        # Get indices of two most similar concepts
        most_similar_indices = sorted(range(len(full_concepts)), key=lambda i: similarity[i], reverse=True)[:2]
        # Add the most similar concepts to the matches list
        matches.append([full_concepts[i] for i in most_similar_indices])

    return matches



output_file = 'TutorQA_test/M1_4_new_batch_clean.tsv'
output_concept_file = 'TutorQA_test/M1_4_new_batch_concepts_top2.tsv'
count = 0
with open(output_file, 'r', newline='', encoding='utf-8') as file, open(output_concept_file,'w') as w:
    reader = csv.reader(file, delimiter='\t')
    # Reading and printing each row
    for row in reader:

        count += 1

        pred_concepts = [x.lower() for x in row[-1].split(';')]
        matched_concepts = match_concepts(pred_concepts)
        flattened_set = list(set(concept for sublist in matched_concepts for concept in sublist))
        print(pred_concepts)
        print(flattened_set)
        print ('----')
        w.write(';'.join(flattened_set))
        w.write('\n')
        # import pdb;
        #
        # pdb.set_trace()
        # pass
        print (count)



