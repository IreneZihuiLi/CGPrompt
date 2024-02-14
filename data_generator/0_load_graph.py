import networkx as nx
import csv

data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
label_path = data_path + 'final_new_annotation.csv'
# label_path = data_path + 'split/train_edges_positive_0.txt'
# test_path = data_path + 'splittrain_edges_positive_0.txt'

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

        if len(row)>1 and row[-1] is '1':
            # import pdb;pdb.set_trace()
            # source, target = map(int, row)
            source = int(row[0])-1
            target = int(row[1]) - 1
            G.add_edge(source, target)

def find_ancestors_by_name(concept_name):

    try:
        # Get the concept ID from its name
        if concept_name not in name_to_id:
            return f"No concept found for '{concept_name}'"
        concept_id = name_to_id[concept_name]

        # Find ancestors of the given concept ID
        ancestors = nx.ancestors(G, concept_id)

        # Return the names of the ancestors
        # return {id: concept_data[id] for id in ancestors}

        ancestor_list =  [concept_data[id] for id in ancestors]
        if len(ancestor_list) >= 1:
            return ancestor_list
    except:
        return None


def find_predecessors_by_name(concept_name):
    try:
        # Get the concept ID from its name
        if concept_name not in name_to_id:
            return f"No concept found for '{concept_name}'"
        concept_id = name_to_id[concept_name]

        # Find predecessors of the given concept ID
        predecessors = G.predecessors(concept_id)

        # Return the names of the predecessors
        # return {id: concept_data[id] for id in predecessors}
        pred_list = [concept_data[id] for id in predecessors]
        if len(pred_list) >=1:

            return pred_list
    except:
        print ('Failed..')


def has_shortest_path_by_name(concept_name_1, concept_name_2):
    # Get the concept IDs from their names
    if concept_name_1 not in name_to_id or concept_name_2 not in name_to_id:
        return "One or both concept names not found"
    concept_id_1 = name_to_id[concept_name_1]
    concept_id_2 = name_to_id[concept_name_2]

    # Check if there is a path between the two concept IDs
    return nx.has_path(G, concept_id_1, concept_id_2)



# Example usage
concept_name = "semantic similarity"  # Replace with the concept name you want to check
ancestors = find_ancestors_by_name(concept_name)
print("Ancestors of Concept:", concept_name, ":", ancestors)


# Example usage
concept_name = "semantic similarity"  # Replace with the concept name you want to check
predecessors = find_predecessors_by_name(concept_name)
print("Predecessors of Concept:", concept_name, ":", predecessors)


# Check shortest path by names
concept_name_1 = "semantic similarity"  # Replace with the first concept name
concept_name_2 = "syntax"  # Replace with the second concept name
path_exists = has_shortest_path_by_name(concept_name_1, concept_name_2)
print("Is there a path from", concept_name_1, "to", concept_name_2, ":", path_exists)




import random

def find_random_path_of_length(graph, path_length):
    if path_length <= 0:
        return "Path length must be a positive integer"

    nodes = list(graph.nodes())
    random.shuffle(nodes)  # Shuffle the nodes to start with a random node

    for start_node in nodes:
        for target_node in nodes:
            if start_node != target_node:
                try:
                    path = nx.shortest_path(graph, source=start_node, target=target_node)
                    if len(path) - 1 == path_length:  # Check if the path length is as desired
                        return path
                except nx.NetworkXNoPath:
                    continue

    return None

# Example usage
path_length = 4  # Length of the path
random_path = find_random_path_of_length(G, path_length)
print("Random Path of Length", path_length, ":", random_path)
print ([concept_data[id] for id in random_path])

print ('Demo finished here..')


import random

def sample_pair(graph, num_samples=50):
    # Ensure we have enough nodes
    if len(graph.nodes()) < 2:
        raise ValueError("Graph must contain at least 2 nodes")

    linked_pairs = []
    unlinked_pairs = []

    # Sample linked pairs
    while len(linked_pairs) < num_samples:
        # Randomly select an edge
        edge = random.choice(list(graph.edges()))
        if edge not in linked_pairs:
            linked_pairs.append(edge)

    # Sample unlinked pairs
    while len(unlinked_pairs) < num_samples:
        # Randomly select two different nodes
        node1, node2 = random.sample(graph.nodes(), 2)
        if not graph.has_edge(node1, node2) and (node1, node2) not in unlinked_pairs:
            unlinked_pairs.append((node1, node2))
    return linked_pairs, unlinked_pairs

def sample_random_linked_concpets(graph,total_paris=50):

    result_pairs = set()
    while len(result_pairs)<total_paris:
        length = random.randint(2,6)
        res = find_random_path_of_length(graph, length)
        result_pairs.add((res[0],res[-1]))

    return result_pairs

## 0115: yes or no question
# _, unlinked_pairs = sample_pair(G,50)
# linked_pairs = sample_random_linked_concpets(G,50)
# with open('TutorQA/M2_1_binary_2_6.tsv', 'w') as w:
#     for (h,t) in linked_pairs:
#         print("Current pair", concept_data[h], ",", concept_data[t])
#         writing_stream = 'In the domain of natural language processing, I already learned about '+concept_data[h]+', based on this, does it help for me to learn about '+concept_data[t]+'?\t'
#         writing_stream += 'Yes'
#         writing_stream += '\n'
#         w.write(writing_stream)
#     for (h,t) in unlinked_pairs:
#         # print("Current pair", concept_data[h], ":", concept_data[t])
#         writing_stream = 'In the domain of natural language processing, I already learned about '+concept_data[h]+', based on this, does it help for me to learn about '+concept_data[t]+'?\t'
#         writing_stream += 'No'
#         writing_stream += '\n'
#         w.write(writing_stream)
#
#
# import pdb;pdb.set_trace()

## 1226, start simple question

# random sample 
with open('TutorQA/M2_2_one_hop.tsv','w') as w:
    count = 0
    special_count = 0
    chosen_concepts = []
    while count < 110:
        query_name = random.choice([x for x in name_to_id.keys()])
        if query_name not in chosen_concepts:
            chosen_concepts.append(query_name)
            ancestors = find_ancestors_by_name(query_name)
            if ancestors is not None and special_count <=54:
                if 'natural language processing intro' in ancestors:
                    special_count += 1
                print(count, query_name)
                print("Ancestors of Concept:", query_name, ":", ancestors)
                writing_stream = 'In the domain of natural language processing, I want to learn about '+query_name+', what concepts should I learn first?\t'
                writing_stream += ';'.join(ancestors)
                writing_stream += '\n'
                w.write(writing_stream)
                count += 1

# with open('TutorQA/M1_one_hop_pds.tsv','w') as w:
#     count = 0
#     while count < 50:
#         query_name = random.choice([x for x in name_to_id.keys()])
#         print (query_name)
#         predecessors = find_predecessors_by_name(query_name)
#         if predecessors is not None:
#             print("Predecessors of Concept:", query_name, ":", predecessors)
#             writing_stream = 'In the domain of natural language processing, I already learned about '+query_name+', based on this, what concepts can I learn in the next?\t'
#             writing_stream += ';'.join(predecessors)
#             writing_stream += '\n'
#             w.write(writing_stream)
#             count += 1

# updated 20240121
# from collections import defaultdict
# # multi hop
# with open('TutorQA/M2_3_multi_hop.tsv','w') as w:
#     count = 0
#     path_set = list()
#     head_count = defaultdict(int)
#     while count < 110:
#         path_length = random.randint(3,7) # Length of the path
#         random_path = find_random_path_of_length(G, path_length)
#         if random_path is not None and random_path not in path_set:
#             head_count[random_path[0]] += 1
#             if head_count[random_path[0]] <= 7:
#                 path_set.append(random_path)
#
#                 print("Random Path of Length", path_length, ":", random_path)
#                 print ("Count ",len(path_set))
#                 concpets = [concept_data[id] for id in random_path]
#                 writing_stream = 'In the domain of natural language processing, I know about '+concpets[0]+', now I want to learn about '+ concpets[-1]+ ', what concept path should I follow?\t'
#                 writing_stream += ';'.join(concpets[1:-1])
#                 writing_stream += '\t'
#                 writing_stream += str(len(concpets)-2)
#                 writing_stream += '\n'
#                 w.write(writing_stream)
#                 count += 1
