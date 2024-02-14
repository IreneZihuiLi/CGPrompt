
import networkx as nx
import csv
import random
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


# random sample some concepts
sampled_items = random.sample(concept_list, 10)
print (sampled_items)



