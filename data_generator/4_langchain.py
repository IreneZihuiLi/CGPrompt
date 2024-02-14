# https://python.langchain.com/docs/use_cases/question_answering/

from langchain.indexes import GraphIndexCreator
from langchain_openai import OpenAI
import csv
import networkx as nx
from langchain_community.graphs.networkx_graph import NetworkxEntityGraph



def load_graph():
    data_path = '../concept_data/'
    concept_path = data_path + '322topics_final.tsv'
    label_path = data_path + 'split/train_edges_positive_0.txt'
    # test_path = data_path + 'splittrain_edges_positive_0.txt'

    # load concept
    concept_data = {}  # Initialize an empty dictionary

    # load concept as dict
    with open(concept_path, 'r') as file:
        for line in file:
            # Split each line at the pipe character
            key, value = line.strip().split('|')
            # Convert key to an integer and strip any whitespace from the value
            concept_data[int(key) - 1] = value.strip()
    # Create a reverse mapping from concept name to ID
    name_to_id = {name: id for id, name in concept_data.items()}

    print(len(concept_data), " concepts loaded.")

    # Create a directed graph
    G = nx.DiGraph()

    # Read relations from CSV and add edges to the graph
    with open(label_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            source, target = map(int, row)
            G.add_edge(source, target)
    return G

concept_graph = load_graph()
print (type(concept_graph))
concept_graph = NetworkxEntityGraph(concept_graph)
print (type(concept_graph))
# import pdb;pdb.set_trace()
# pass

# from langchain.chains import GraphQAChain
# chain = GraphQAChain.from_llm(OpenAI(temperature=0,openai_api_key="sk-"), graph=concept_graph, verbose=True)
# question = 'Use the graph, which defines the prerequisite relations, then answer the question using the nodes from the given graph: in the domain of natural language processing, I know about Markov chains, now I want to learn about social media analysis, what concept path should I follow?'
# print (chain.invoke(question))

'''
> Entering new GraphQAChain chain...
Entities Extracted:
 Markov chains, social media analysis
Full Context:


> Finished chain.
{'query': 'In the domain of natural language processing, I know about Markov chains, now I want to learn about social media analysis, what concept path should I follow?', 'result': "\nI don't know, as I am a computer program and do not have the ability to provide recommendations or suggestions. It would be best to consult with a human expert in the field of natural language processing for guidance on what concepts to learn next for social media analysis."}

'''





