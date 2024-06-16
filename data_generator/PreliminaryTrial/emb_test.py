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

# Example usage
pred_list = ['parsing evaluation', 'feature learning', 'neural parsing', 'variational bayes models']
gold_list = ['syntax analysis', 'attribute learning', 'deep parsing techniques', 'bayesian methods']
print(match_embedding(pred_list, gold_list))
