import os
import pickle
import statistics as stats

with open('../save/graph.pkl', 'rb') as f:
    G = pickle.load(f)

concept_count = sum(1 for _, data in G.nodes(data=True) if data.get('type') == 'concept')
# print(concept_count)

for node, data in G.nodes(data=True):
    if data['type'] == 'concept':
        # print(node)
        # print(data['text'])
        break

print ('start')


def filter_doc_by_length(doc_string, token_num=150):
    new_string = doc_string.split(' ')[:token_num]
    return ' '.join(new_string)

def get_node_name(concept_id):
    concept_id+=1
    return G.nodes[f'c{concept_id}']['text']

def get_neightbors(concept_id,neighbor_num=8):
    concept_id += 1
    # return in list of strings
    neighbor_list=[]
    neighbor_doc=[]
    for neigh_doc in G.neighbors(f'c{concept_id}'):
        neighbor_list.append(neigh_doc)
        # neighbor_doc.append(G.nodes[neigh_doc]['text'])
        neighbor_doc.append(filter_doc_by_length(G.nodes[neigh_doc]['text']))

    # return neighbor_list[:neighbor_num],neighbor_doc[:neighbor_num]
    doc_stream="<doc>"
    doc_stream+="</doc> <doc>".join(neighbor_doc[:neighbor_num])
    doc_stream+="</doc>"
    return neighbor_list[:neighbor_num],neighbor_doc[:neighbor_num],doc_stream

def count_prompt_length(prompt_str):
    # Splitting the string by spaces to get the tokens
    tokens = prompt_str.split()
    # Returning the number of tokens
    return len(tokens)


# print (get_node_name(1))
# ngb_list,ngb_doc,doc_stream = get_neightbors(1,2)
# print (ngb_list)



# start template generation

instruction_path='../instruction_test/1205/'
data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
annotation_positive = 'split/test_edges_positive_'
annotation_negative = 'split/test_edges_negative_'

# load concept
concept_data = {}  # Initialize an empty dictionary

# load concept as dict
with open(concept_path, 'r') as file:
    for line in file:
        # Split each line at the pipe character
        key, value = line.strip().split('|')
        # Convert key to an integer and strip any whitespace from the value
        concept_data[int(key)-1] = value.strip()

print (len(concept_data)," concepts loaded.")


# t1:
# template_head="provide \"prerequisite or dependency\" " \
#          "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
#               "learning A would help people to learn B, note this relation is directional, " \
#           "which means (B,A) is false but (A,B) is true. Is there such a relation between (" \
#           # "(autoencoder,variational autoencoder" \\
# template_mid=")? And there are related content to help: "
# template_tail=" TELL ME YES OR NO ONLY. Use your own knowledge.\n"

# t2:
# template_head="provide \"prerequisite or dependency\" " \
#          "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
#               "learning A would help people to learn B, note this relation is directional, " \
#           "which means (B,A) is false but (A,B) is true. If learning A can only help a little to learn B, then you should answer NO. Is there such a relation between (" \
#           # "(autoencoder,variational autoencoder" \\
# template_mid=")? And there are related content to help: "
# template_tail=" TELL ME YES OR NO ONLY. Use your own knowledge.\n"

# t3,t4
template_head="provide \"prerequisite or dependency\" " \
         "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
              "learning A would help people to learn B, note this relation is directional, " \
          "which means (B,A) is false but (A,B) is true. If there is no strong or directed relation, then you should answer NO. Is there such a relation between (" \
          # "(autoencoder,variational autoencoder" \\
template_mid=")? And there are related content to help: "
template_tail=" TELL ME YES OR NO ONLY. Use your own knowledge.\n"
# import pdb;pdb.set_trace()



def generate_instruction(batch_id):


    batch_id = str(batch_id)
    label_path_neg = data_path + annotation_negative + batch_id + '.txt'
    label_path_pos = data_path + annotation_positive + batch_id + '.txt'

    #count avg token length
    token_length=[]

    os.makedirs(instruction_path, exist_ok=True)
    with open(instruction_path+'t3.neg.'+batch_id+'.txt','w') as writer:
        with open(label_path_neg,'r') as file:
            for line in file:
                source, target=line.strip().split(',')
                source_ngb_list, source_ngb_doc, source_stream = get_neightbors(int(source))
                target_ngb_list, target_ngb_doc, target_stream = get_neightbors(int(target))
                prompt = template_head+concept_data[int(source)]+' , '+concept_data[int(target)]+template_mid + source_stream + target_stream + template_tail
                # print (prompt)
                writer.write(prompt)
                token_length.append(count_prompt_length(prompt))

    with open(instruction_path+'t3.pos.'+batch_id+'.txt','w') as writer:
        with open(label_path_pos,'r') as file:
            for line in file:
                source, target = line.strip().split(',')
                source_ngb_list, source_ngb_doc, source_stream = get_neightbors(int(source))
                target_ngb_list, target_ngb_doc, target_stream = get_neightbors(int(target))
                prompt = template_head + concept_data[int(source)] + ' , ' + concept_data[
                    int(target)] + template_mid + source_stream + target_stream + template_tail
                token_length.append(count_prompt_length(prompt))
                writer.write(prompt)
    print (batch_id)
    print("Token Length:")
    print("Max {}, Min {}, Mean {}, Median {}".format(max(token_length), min(token_length), stats.mean(token_length),
                                                      stats.median(token_length)))


for i in range(5):
    generate_instruction(i)



'''
t1:

t2:
322  concepts loaded.
0
Token Length:
Max 1687, Min 85, Mean 1324.2903225806451, Median 1675.5
1
Token Length:
Max 1687, Min 86, Mean 1285.6483870967743, Median 1486.0
2
Token Length:
Max 1687, Min 86, Mean 1339.5032258064516, Median 1683.0
3
Token Length:
Max 1688, Min 86, Mean 1312.8967741935485, Median 1535.5
4
Token Length:
Max 1688, Min 86, Mean 1304.8, Median 1647.5



t3:
0
Token Length:
Max 1684, Min 82, Mean 1321.2903225806451, Median 1672.5
1
Token Length:
Max 1684, Min 83, Mean 1282.6483870967743, Median 1483.0
2
Token Length:
Max 1684, Min 83, Mean 1336.5032258064516, Median 1680.0
3
Token Length:
Max 1685, Min 83, Mean 1309.8967741935485, Median 1532.5
4
Token Length:
Max 1685, Min 83, Mean 1301.8, Median 1644.5


t3/1205:[400, bad request]
Token Length:
Max 3284, Min 82, Mean 2414.374193548387, Median 2673.5
1
Token Length:
Max 3284, Min 83, Mean 2337.8709677419356, Median 2282.5
2
Token Length:
Max 3284, Min 83, Mean 2446.877419354839, Median 2682.0
3
Token Length:
Max 3285, Min 83, Mean 2360.877419354839, Median 2326.5
4
Token Length:
Max 3285, Min 83, Mean 2370.8709677419356, Median 2481.0


t3/1205: [ ok ]
Token Length:
Max 2484, Min 82, Mean 1833.3516129032257, Median 2031.0
1
Token Length:
Max 2484, Min 83, Mean 1775.683870967742, Median 1732.5
2
Token Length:
Max 2484, Min 83, Mean 1857.6709677419356, Median 2032.0
3
Token Length:
Max 2485, Min 83, Mean 1793.3, Median 1776.5
4
Token Length:
Max 2485, Min 83, Mean 1800.1967741935484, Median 1881.0

'''

