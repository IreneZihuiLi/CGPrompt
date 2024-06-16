# -*- coding: utf-8 -*-

# !pip install --upgrade openai
from google.colab import drive
drive.mount('/content/drive')

"""# Extract concept list"""

g2_file_path = 'concept_abstracts_70.json'
import json
with open(g2_file_path) as f:
    data = json.load(f)
concept_list = [x for x in data.keys()]
print (len(concept_list))

def get_background(concept_name):
  background = data[concept_name]['abstracts']
  background = ' '.join(background)
  return background
get_background('spelling correction')

"""# Extract 322 expert concept graph"""

import networkx as nx
import csv
import random
data_path = 'concept_data/'
concept_path = data_path + '322topics_final.tsv'
label_path = data_path + 'final_new_annotation.csv'


# load concept
concept_data = {}  # Initialize an empty dictionary


# load concept as dict
with open(concept_path, 'r') as file:
    for line in file:
        # Split each line at the pipe character
        key, value = line.strip().split('|')
        # Convert key to an integer and strip any whitespace from the value
        concept_data[int(key)-1] = value.strip()
# Create a reverse mapping from concept name to ID
name_to_id = {name: id for id, name in concept_data.items()}

print (len(concept_data)," concepts loaded.")


# Create a directed graph
G = nx.DiGraph()

# Read relations from CSV and add edges to the graph
with open(label_path, 'r') as file:
    reader = csv.reader(file)
    for row in reader:

        if len(row)>1 and row[-1] == '1':
            # import pdb;pdb.set_trace()
            # source, target = map(int, row)
            source = int(row[0])-1
            target = int(row[1]) - 1
            G.add_edge(source, target)

def get_neighbors(concept_name):
    try:
        # Get the concept ID from its name
        if concept_name not in name_to_id:
            return []
        concept_id = name_to_id[concept_name]

        # Get all neighbors (successors and predecessors)
        neighbors = set(G.predecessors(concept_id)).union(set(G.successors(concept_id)))

        # Return the names of the neighbors
        neighbor_list = [concept_data[id] for id in neighbors]
        return neighbor_list
    except Exception as e:
        return []

def get_2hop_neighbors(concept_name):
    try:
        # Get the concept ID from its name
        if concept_name not in name_to_id:
            return []
        concept_id = name_to_id[concept_name]

        # Get 1-hop neighbors (successors and predecessors)
        neighbors_1hop = set(G.predecessors(concept_id)).union(set(G.successors(concept_id)))

        # Initialize a set for 2-hop neighbors
        neighbors_2hop = set()

        # Find 2-hop neighbors by looking at neighbors of 1-hop neighbors
        for neighbor in neighbors_1hop:
            neighbors_2hop.update(set(G.predecessors(neighbor)))
            neighbors_2hop.update(set(G.successors(neighbor)))

        # Remove the original concept and 1-hop neighbors from the 2-hop neighbors set
        neighbors_2hop.discard(concept_id)
        neighbors_2hop -= neighbors_1hop

        # Return the names of the 2-hop neighbors
        neighbor_list = [concept_data[id] for id in neighbors_2hop]
        return neighbor_list
    except Exception as e:
        return []

def get_out_neighbors(concept_name):
    # Check if the concept name exists in the mapping
    if concept_name not in name_to_id:
        return []
    try:
      # Get the concept ID from its name
      concept_id = name_to_id[concept_name]

      # Find all outgoing neighbors
      out_neighbors = list(G.successors(concept_id))

      # Convert neighbor IDs back to concept names
      neighbor_list = [concept_data[id] for id in out_neighbors]
      return neighbor_list
    except Exception as e:
      return []

def get_in_neighbors(concept_name):
    # Check if the concept name exists in the mapping
    if concept_name not in name_to_id:
        return []
    try:
      # Get the concept ID from its name
      concept_id = name_to_id[concept_name]

      # Find all incoming neighbors
      in_neighbors = list(G.predecessors(concept_id))

      # Convert neighbor IDs back to concept names
      in_neighbor_names = [concept_data[neighbor_id] for neighbor_id in in_neighbors]

      return in_neighbor_names
    except Exception as e:
      return []

# Test the function
print(get_neighbors('named entity recognition'))


# Test the function
print(get_2hop_neighbors('named entity recognition'))

print(get_out_neighbors('natural language processing intro'))
print(get_in_neighbors('named entity recognition'))

def get_graph2(concept_name):
    g2_neighbors = get_out_neighbors(concept_name)
    if len(g2_neighbors) >= 1:
        res_str = ''
        g2_item_template = "({head},Is-a-Prerequisite-of,{tail})\n"
        for neighbor in g2_neighbors:
            formatted_item = g2_item_template.format(head=concept_name, tail=neighbor)
            res_str += formatted_item
        return res_str
    else:
        return "None"


# Test the function
print(get_graph2('natural language processing intro'))
print(get_graph2('named entity recognition'))

"""# Extract LLM triplets"""

llm_file_path = ''
import json,re
import ast

# model_name='llama3_70b'
# model_name="gpt-4-turbo"
model_name= 'gpt-4o'
# model_name='gpt-3.5-turbo'

# read from the res files
triplets = []


with open(llm_file_path+model_name+'-triplets.txt', 'r') as f:
    for line in f:
        # Strip any leading/trailing whitespace characters and convert the string to a tuple
        triplet = ast.literal_eval(line.strip())
        triplets.append(triplet)
print (len(triplets))
triplets[:4]

def get_graph1(concept_name):
  res_str = ''
  g1_item_template = "({head},{relation},{tail})\n"
  for triplet in triplets:
    if concept_name == triplet[0] or concept_name == triplet[2]:
      content = g1_item_template.format(head=triplet[0], relation=triplet[1], tail=triplet[2])
      res_str += content
  return res_str

# Test the function
concept_name = 'named entity recognition'
# Test the function
concept_name = 'paraphrasing'
print(get_graph1(concept_name))

"""# Now start prompting"""

import os
os.environ['OPENAI_API_KEY'] = 'sk-'
from openai import OpenAI
client = OpenAI()

instruction = '''
###Instruction: You are a knowledge graph builder. Now please fuse two sub-knowledge graphs about the concept "{concept}".
Graph 1:
{graph1}

Graph 2:
{graph2}

Rules for Fusing the Graphs:
1. Union the concepts and edges.
2. If two concepts are similar, or they are referring to the same concept, combine them as one concept by keeping the meaningful or specific one. For example, "lstm" versus "long short-term memory",  please keep "long short-term memory".
3. We only allow one relation to exist between two concepts, if there is a conflict,  read the following "##background" to help you keep the correct one. knowledge to keep the correct one.  For example, (ROUGE, Evaluate-for, question answering model) and (ROUGE,Used-for , question answering model) are considered to be conflicts.
4. Once step 3 is done, consider every possible concept pair, which did not occur in step 2. For example, take a concept in G1, and match a concept from G2. And look at the "##background" ,  and summarize new triplets.

Hint: the relation types and their definition. You can use it to do Step 3:
   a) Compare: Represents a relationship between two or more entities where a comparison is being made. For example, "A is larger than B" or "X is more efficient than Y."
   b) Part-of: Denotes a relationship where one entity is a constituent or component of another. For instance, "Wheel is a part of a Car."
   c) Conjunction: Indicates a logical or semantic relationship where two or more entities are connected to form a group or composite idea. For example, "Salt and Pepper."
   d) Evaluate-for: Represents an evaluative relationship where one entity is assessed in the context of another. For example, "A tool is evaluated for its effectiveness."
   e) Is-a-Prerequisite-of: This dual-purpose relationship implies that one entity is either a characteristic of another or a required precursor for another. For instance, "The ability to code is a prerequisite of software development."
   f) Used-for: Denotes a functional relationship where one entity is utilized in accomplishing or facilitating the other. For example, "A hammer is used for driving nails."
   g) Hyponym-Of: Establishes a hierarchical relationship where one entity is a more specific version or subtype of another. For instance, "A Sedan is a hyponym of a Car."


##Background: {background}

###Output Instruction: Output the new merged data by listing the triplets.	Your answer should ONLY contain triplets in this format: (concept, relation, concept). No other explanations or numbering are needed. Only triplets, no intermediate results.
'''

def get_answer_GPT(instruction):
    completion = client.chat.completions.create(
      # model=model_name,
      model='gpt-4o',
      messages=[
        {"role": "system", "content": "You are a knowledge graph builder."},
        {"role": "user", "content": instruction}
      ],
      max_tokens=300  # Sets the maximum length of the response
    )

    # Ensure that the response is correctly accessed
    if completion.choices:
        # print(completion.choices[0].message.content)
        answer = completion.choices[0].message.content.replace('\n','')
    else:
        print("No completion found.")
        answer = 'ERROR'

    return answer

#testing
print (len(concept_list))
tmp_list = concept_list


# start
empty_count = 0
fusion_res_path = llm_file_path+model_name+'-fusion.txt'
with open(fusion_res_path,'a') as w:
  for id,test_concept in enumerate(tmp_list):
    g1 = get_graph1(test_concept)
    bg = 'no background'
    # if g1 is empty, skip
    if len(g1) >= 3:
      g2 = get_graph2(test_concept)
      bg = get_background(test_concept)[:15000]
      prompt = instruction.format(concept=test_concept, graph1=g1, graph2=g2, background=bg)


      try:
        # Get the answer from GPT
        answer = get_answer_GPT(prompt)
      except Exception as e:
          print("error")
          print (e)
          answer = "error"
    else:
      empty_count += 1
      answer = "EMPTY G1." + test_concept
    print(id, len(bg))
    print (answer)
    w.write(answer.replace('\n','. ')+'\n')
    w.flush()

print (empty_count)

"""# Clean results

"""

relation_list = ['Compare','Part-of','Conjunction','Evaluate-for','Is-a-Prerequisite-of','Used-for','Hyponym-Of']
fusion_res_path = llm_file_path+model_name+'-fusion.txt'
with open(fusion_res_path) as f:
    data = f.read()
# Extract all triplets using regex
triplet_pattern = re.compile(r'\(([^)]+)\)')
triplets = triplet_pattern.findall(data)


filtered_triplets = []
for triplet in triplets:
    triplet = triplet.replace('<','').replace('>','')
    parts = triplet.split(',')
    if len(parts)==3:
        relation = parts[1].strip().replace('"', '').replace("'", '')
        if relation in relation_list:
            item = (
                parts[0].strip().replace('"', '').replace("'", ''),
                relation,
                parts[2].strip().replace('"', '').replace("'", '')
            )
            filtered_triplets.append(item)


print (len(filtered_triplets))

relation_count = {relation: 0 for relation in relation_list}
for triplet in filtered_triplets:
    relation = triplet[1]
    # print (relation)
    # print (triplet)
    relation_count[relation] += 1

# Print the results
print("Filtered triplets:")
print(filtered_triplets)
print("\nRelation statistics:")
# for relation, count in relation_count.items():
#     print(f"{relation}: {count}")
stat_res_path = llm_file_path+model_name+'-fusion-clean-stats.txt'
with open(stat_res_path,'a') as w:
  for relation, count in relation_count.items():
    print(f"{relation}: {count}")
    w.write(f"{relation}: {count}\n")
final_res_path = llm_file_path+model_name+'-fusion-triplets.txt'
with open(final_res_path,'a') as w:
    for item in filtered_triplets:
        w.write(str(item)+'\n')