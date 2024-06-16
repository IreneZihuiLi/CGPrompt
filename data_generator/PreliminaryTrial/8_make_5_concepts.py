import networkx as nx
import csv
import pdb,random

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

        if len(row)>1 and row[-1] == '1':
            # import pdb;pdb.set_trace()
            # source, target = map(int, row)
            source = int(row[0]) - 1
            target = int(row[1]) - 1
            G.add_edge(source, target)

print ('Graph loaded')


def sample_neighborhood_with_resampling(G, hops=3, node_num=4):
    while True:
        # Step 1: Randomly sample a node K
        K = random.choice(list(G.nodes()))

        # Step 2: Find the neighborhood of K within 3 hops
        neighbors_within_hops = set([K])
        to_visit = [K]

        for _ in range(hops):
            next_visit = []
            for node in to_visit:
                for neighbor in G.neighbors(node):
                    neighbors_within_hops.add(neighbor)
                    next_visit.append(neighbor)
            to_visit = next_visit

        # Check if the neighborhood has at least 3 nodes, resample if not
        if len(neighbors_within_hops) >= node_num:
            break

    # Step 3: Randomly sample 3 nodes from the neighborhood, including node K
    sampled_nodes = random.sample(neighbors_within_hops, node_num)

    return sampled_nodes


# test code
sampled_ids = sample_neighborhood_with_resampling(G,hops=3)
print ([concept_data[id] for id in sampled_ids])


def sample_neighborhoods_multiple_times(G, num_samples=150, hops=3):
    unique_samples = set()

    while len(unique_samples) < num_samples:
        node_num = random.randint(3,5)
        sampled_nodes = sample_neighborhood_with_resampling(G, hops,node_num)
        unique_samples.add(tuple(sorted(sampled_nodes)))  # Sorting to ensure consistency in comparison

    return list(unique_samples)
final_id_set=list(sample_neighborhoods_multiple_times(G))
print ('Sampling finished\n Now Generating data')
header = 'In the domain of natural language processing, what potential project can I work on? Give me a possible idea. Show me title, project description (around 150 tokens). '
with open('TutorQA_test/M1_5_idea_100.tsv','w') as w:
    w.write('Concepts\tQuestion\n')
    for item in final_id_set:
        concepts = [concept_data[id] for id in item]
        writing = ';'.join(concepts)
        writing += '\t'
        question = 'I already know about '
        question +=' , '.join(concepts[:-1])
        question += ' and '
        question += concepts[-1]
        question = question + '.' + header
        writing+=question
        w.write(writing+'\n')
print ('Done')





#
pdb.set_trace()
# pass
