'''
this is to load concept name and data;
Note: the concept ID need to -1
'''

import os

"""  
modify by sixun
changed: the paths for different domain: NLP, CV, BIO
"""
domain = 'NLP' # change here to switch the domain
if domain == 'NLP':
    instruction_path='../instruction_test/1205/'
    data_path = '../concept_data/'
    concept_path = data_path + '322topics_final.tsv'
    annotation_positive = 'split/test_edges_positive_'
    annotation_negative = 'split/test_edges_negative_'

elif domain == 'CV':
    instruction_path='../instruction_test/CV/'
    data_path = '../concept_data/'
    concept_path = data_path + 'CV_topics.tsv'
    annotation_positive = 'CV_split/test_pos_'
    annotation_negative = 'CV_split/test_neg_'

elif domain == 'BIO':
    instruction_path='../instruction_test/BIO/'
    data_path = '../concept_data/'
    concept_path = data_path + 'BIO_topics.tsv'
    annotation_positive = 'BIO_split/test_pos_'
    annotation_negative = 'BIO_split/test_neg_'


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

# template 1
# template_head="provide \"prerequisite or dependency\" " \
#          "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
#               "learning A would help people to learn B, note this relation is directional, " \
#           "which means (B,A) is false but (A,B) is true. Is there such a relation between (" \
#           # "(autoencoder,variational autoencoder" \
# # template_tail=")? TELL ME YES OR NO ONLY. Use your own knowledge.\n"
# template_tail=")? TELL ME YES OR NO ONLY. Use your own knowledge.\n"
# import pdb;pdb.set_trace()


# template 11
template_head="provide \"prerequisite or dependency\" " \
         "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
              "learning A would help people to learn B, note this relation is directional, " \
          "which means (B,A) is false but (A,B) is true. If there is no strong or directed relation, then you should answer NO. Is there such a relation between (" \
          # "(autoencoder,variational autoencoder" \
# template_tail=")? TELL ME YES OR NO ONLY. Use your own knowledge.\n"
template_tail=")? TELL ME YES OR NO ONLY. Use your own knowledge.\n"


# load label
def generate_instruction(batch_id):
    batch_id = str(batch_id)
    label_path_neg = data_path + annotation_negative + batch_id + '.txt'
    label_path_pos = data_path + annotation_positive + batch_id + '.txt'

    os.makedirs(instruction_path, exist_ok=True)
    with open(instruction_path+'t11.neg.'+batch_id+'.txt','w') as writer:
        with open(label_path_neg,'r') as file:
            for line in file:
                source, target=line.strip().split(',')
                # print (concept_data[int(source)],concept_data[int(target)])
                writer.write(template_head+concept_data[int(source)]+' , '+concept_data[int(target)]+template_tail)

    with open(instruction_path+'t11.pos.'+batch_id+'.txt','w') as writer:
        with open(label_path_pos,'r') as file:
            for line in file:
                source, target=line.strip().split(',')
                # print (concept_data[int(source)],concept_data[int(target)])
                writer.write(template_head+concept_data[int(source)]+' , '+concept_data[int(target)]+template_tail)
                # import pdb;pdb.set_trace()
                # pass

for i in range(5):
    generate_instruction(i)






