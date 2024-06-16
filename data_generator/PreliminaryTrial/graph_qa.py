# !pip install neo4j langchain langchain_openai -q

from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from tqdm import tqdm
uri = "bolt://localhost:7687"
username = "neo4j"
password = "12345678"


data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
label_path = data_path + 'split/train_edges_positive_0.txt'


driver = GraphDatabase.driver(uri, auth=(username, password))

def create_concepts(tx, concepts):
    for concept_id, concept_name in concepts:
        tx.run("MERGE (:Concept {concept_id: $id, concept_name: $name})", id=concept_id, name=concept_name)

def create_relationships(tx, relationships):
    for source_id, target_id in relationships:
        tx.run(
            "MATCH (a:Concept {concept_id: $source_id}), (b:Concept {concept_id: $target_id}) "
            "MERGE (a)-[:PREREQUISITE]->(b)",
            source_id=source_id, target_id=target_id
        )

def import_concepts(tsv_path):
    concepts = []
    with open(tsv_path, "r") as file:
        for line in file:
            parts = line.strip().split('|')
            concept_id = int(parts[0].strip())
            concept_name = parts[1].strip()
            concepts.append((concept_id, concept_name))

        print ('Concepts loaded.')
    with driver.session() as session:
        session.execute_write(create_concepts, concepts)




def import_relationships(txt_path):
    relationships = []
    with open(txt_path, "r") as file:
        for line in file:
            parts = line.strip().split(",")
            if len(parts) == 2:
                source_id = int(parts[0].strip()) + 1
                target_id = int(parts[1].strip()) + 1
                relationships.append((source_id, target_id))
        print('Relation loaded.')
    with driver.session() as session:
        session.execute_write(create_relationships, relationships)


# topics_tsv_path = concept_path
# import_concepts(topics_tsv_path)
# edges_txt_path = label_path
# import_relationships(edges_txt_path)


from langchain.chains import GraphCypherQAChain

graph = Neo4jGraph(
    url=uri, username=username, password=password
)

graph.refresh_schema()
print(graph.schema)

from langchain.chat_models import ChatOpenAI

import os

os.environ["OPENAI_API_KEY"] = "sk-"

chain = GraphCypherQAChain.from_llm(
    # ChatOpenAI(temperature=0, model="gpt-4-1106-preview"),
    ChatOpenAI(temperature=0, model= "gpt-3.5-turbo"),
    graph=graph,
    verbose=True,
    top_k=3
)

one_shot = ''' Return the result in the following format: 1. concept_name 2. concept_name ...
Example:
 A->B
If i want to learn B, I need to learn A first.
If no result or answer is found, use your own knowledge to answer the question.
'''


# Simple testing code
# results = chain.run("""
# In the domain of natural language processing, I want to learn about discourse model, what concepts should I learn first?
# Return the result in the following format: 1. concept_name 2. concept_name ...
#  """+one_shot)
# print(results)

summary_test = '''I will give you a project description, based on it, tell me what concepts should I learn so as to finish it. 
This project revolves around creating an ML model that can detect emotions from the conversations we have commonly in our daily life. The ML model can detect up to five different emotions and offer personalized recommendations based on your present mood.
This emotion-based recommendation engine is of immense value to many industries as they can use it to sell to highly targeted audience and buyer personas. For instance, online content streaming platforms can use this tool to offer customized content suggestions to individuals by reading their current mood and preference. '''
results = chain.run(summary_test)
print(results)



# # begin testing
# instruction_path = 'TutorQA_test/M1_2_one_hop.tsv'
# result_path = 'output_langchain/M2_one_hop_anc.tsv'
# with open(instruction_path,'r') as f, open(result_path,'w') as w:
#     lines = f.readlines()[:5]
#     for line in tqdm(lines, desc="Processing"):
#
#         question, answer = line.split("\t")
#         question = question + one_shot
#         try:
#             #TODO: dead loop may occur, set time
#             res = chain.run(question)
#         except:
#             res = 'Failed, I do not know!'
#
#         res = res.replace('\n', '\t')
#         res = res + " ***** " + answer
#         w.write(res)
#         print(res)
#         import pdb;pdb.set_trace()
#         pass


'''

I want to learn about "part of speech tagging", what concepts should I learn "syntax" first?
Answer Yes or No.  
If there is no such relation exist, the answer is No. 

prompt 1
 Example:
 A->B->C
 If i want to learn C, I need to learn B, and B needs A.
 So I need learn A and B.

prompt 2
Return in json format.
 Example:
 A->B
 If i want to learn B, I need to learn A. 
'''


'''
Failed cases:
In the domain of natural language processing, I know about "matrix multiplication", now I want to learn about "lexicography", what concept path should I follow?
List all possible concepts. If you do not have the answer, use your own knowledge. 
Return in json format. 



one-shot: so many empty answers
Return the result in the following format: 1. concept_name 2. concept_name ...
Example:
 A->B
 If i want to learn B, I need to learn A. 
If no concept or answer is found, use your own knowledge. 
'''

