import pickle
import torch
from openai import OpenAI
client = OpenAI()


with open('../save/graph.pkl', 'rb') as f:
    G = pickle.load(f)

def run_and_store(input_message):
    response = client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-3-small" # 1536
    )
    output = response.data[0].embedding
    return output

def concept_run():
    concept_generated_emb = []

    for concept_id in range(1, 322 + 1):
        concept_text = G.nodes[f'c{concept_id}']['text']
        print(f"c{concept_id}: ", concept_text)

        emb = run_and_store(concept_text)
        concept_generated_emb.append(emb)

    with open('./concept_emb_openai_small.pkl', 'wb') as f:
        pickle.dump(concept_generated_emb, f)

if __name__ == '__main__':
    concept_run()