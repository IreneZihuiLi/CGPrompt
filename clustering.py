from transformers import BertModel, AutoTokenizer, AutoModelForMaskedLM
import torch

tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
#model = BertModel.from_pretrained("data/japanese_med/ja_med_bert/checkpoint-10")
model = AutoModelForMaskedLM.from_pretrained("data/japanese_med/ja_med_bert/checkpoint-10")


def get_embedding(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors='pt')

    # Get embeddings
    with torch.no_grad():
        outputs = model(**inputs)

    # Get the last hidden states
    last_hidden_states = outputs.logits

    # Compute sentence embedding by averaging over token embeddings
    return last_hidden_states.mean(dim=1).squeeze()

concepts = []
embeddings = []

for line in open('data/japanese_med/concepts.tsv', 'r'):
    concept = line.strip()
    concepts.append(concept)
    embeddings.append(get_embedding(model, tokenizer, concept))
