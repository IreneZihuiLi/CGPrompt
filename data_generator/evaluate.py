import csv
import chromadb
import numpy as np
from chromadb.utils import embedding_functions
from sklearn.metrics.pairwise import cosine_similarity


def evaluate(path: str, concept_path: str):
    # res = csv.reader(open(path, 'r'), delimiter='\t')

    # this is for langchain
    with open(path,'r') as f:
        res = f.readlines()

    concept_data = {}
    with open(concept_path, 'r') as file:
        for line in file:
            # Split each line at the pipe character
            key, value = line.strip().split('|')
            # Convert key to an integer and strip any whitespace from the value
            concept_data[int(key) - 1] = value.strip().lower()

    sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
        name="agent_list",
        embedding_function=sentence_transformer_ef,
    )
    collection.add(
        documents=list(concept_data.values()),
        metadatas=[{"source": "concept from knowledge graph"} for _ in range(len(concept_data.values()))],
        ids=[f"concept_{i}" for i in concept_data.keys()],
    )

    res_list = []
    ans_list = []
    for row in res:
        if len(row) == 0:
            continue
        answer = row[0]
        # ans_list.append(row[0])

        # this is for langchain
        row = [x.strip() for x in row.split('*****')]
        if 'the following concepts:\t' in row[0]:
            answer = ' '.join(row[0].split('\t')[1:])

        ans_list.append(answer)
        # import pdb;pdb.set_trace()
        res_ans_list = row[1].split(';')
        res_ans_list = [f"{idx + 1}. {x.strip().lower()}; " for idx, x in enumerate(res_ans_list)]
        res_ans_str = ''.join(res_ans_list)
        res_list.append(res_ans_str)

    # import pdb;
    # pdb.set_trace()

    ans_embedding = sentence_transformer_ef(ans_list)
    res_embedding = sentence_transformer_ef(res_list)

    cos_sim = cosine_similarity(ans_embedding, res_embedding)
    print(np.mean(cos_sim))


if __name__ == "__main__":
    # res_path = 'output_llama/M1_2_one_hop.tsv'
    # res_path = 'output_gpt4/M1_2_one_hop.tsv'
    # res_path = 'output_gpt3/M1_2_one_hop.tsv'
    res_path = 'output_langchain/M1_one_hop_anc.tsv'

    data_path = '../concept_data/'
    concept_path = data_path + '322topics_final.tsv'

    print (res_path)
    evaluate(res_path, concept_path)



'''

'''