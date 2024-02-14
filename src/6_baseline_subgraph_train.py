import os
import pickle
import statistics as stats


domain = 'BIO'
if domain == 'NLP':
    instruction_path='../instruction_test/1205_train/'
    data_path = '../concept_data/'
    concept_path = data_path + '322topics_final.tsv'
    annotation_positive = 'split/test_edges_positive_'
    annotation_negative = 'split/test_edges_negative_'

    train_annotation_positive = data_path+'split/train_edges_positive_'
    train_annotation_negative = data_path+'split/train_edges_negative_'

elif domain == 'CV':
    instruction_path='../instruction_test/CV_train/'
    data_path = '../concept_data/'
    concept_path = data_path + 'CV_topics.tsv'
    annotation_positive = 'CV_split/test_pos_'
    annotation_negative = 'CV_split/test_neg_'

    train_annotation_positive = data_path + 'CV_split/train_pos_'
    train_annotation_negative = data_path + 'CV_split/train_neg_'

elif domain == 'BIO':
    instruction_path='../instruction_test/BIO_train/'
    data_path = '../concept_data/'
    concept_path = data_path + 'BIO_topics.tsv'
    annotation_positive = 'BIO_split/test_pos_'
    annotation_negative = 'BIO_split/test_neg_'

    train_annotation_positive = data_path + 'BIO_split/train_pos_'
    train_annotation_negative = data_path + 'BIO_split/train_neg_'


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

# load train concept
def load_train(batch):
    relations = {}
    reversed_relations={}
    with open(train_annotation_positive+batch+'.txt') as file:
        for line in file:
            # Split each line at the comma and convert to integers
            source, target = map(int, line.strip().split(','))

            # Add the target to the source's list in the dictionary
            if source in relations:
                relations[source].append(target)
            else:
                relations[source] = [target]
            
            # add reversed relations
            if target in reversed_relations:
                relations[target].append(source)
            else:
                relations[target] = [source]

    return relations,reversed_relations

# 1205_train, t1
# template_head="provide \"prerequisite or dependency\" " \
#          "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
#               "learning A would help people to learn B, note this relation is directional, " \
#           "which means (B,A) is false but (A,B) is true. If there is no strong or directed relation, then you should answer NO. Is there such a relation between (" \
#           # "(autoencoder,variational autoencoder" \\
# template_mid=")? Following are some information to help you answer this: "
# template_tail=" TELL ME YES OR NO ONLY. Add your own knowledge.\n"

# 1205_train, t2
template_head="provide \"prerequisite or dependency\" " \
         "relations between these key concepts. The prerequisite relation on two concepts (A,B) or A->B, means, " \
              "learning A would help people to learn B, note this relation is directional, " \
          "which means (B,A) is false but (A,B) is true. If there is no strong or directed relation, then you should answer NO. Is there such a relation between (" \
          # "(autoencoder,variational autoencoder" \\
template_mid=")? Following are some information to help you answer this: "
template_tail=" TELL ME YES OR NO ONLY. Add your own knowledge.\n"


def get_neighbors(concept_ID,relations,reversed_relations):
    neighbor_names = []
    neighor_IDs = []
    reversed_neighbor_names = []
    reversed_neighor_IDs = []
    if concept_ID in relations.keys():
        neighor_IDs+=relations[concept_ID]
        for nb in relations[concept_ID]:
            neighbor_names.append(concept_data[nb])
    if concept_ID in reversed_relations.keys():
        reversed_neighor_IDs+=reversed_relations[concept_ID]
        for nb in reversed_relations[concept_ID]:
            reversed_neighbor_names.append(concept_data[nb])   
    return  neighor_IDs, reversed_neighor_IDs, neighbor_names, reversed_neighbor_names
    
def get_content(src_ID, tgt_ID,relations,reversed_relations):
    # given a pair (src_ID, tgt_ID), return the subgraph and linearize them as content
    _, _, src_neighbor_names, src_reversed_neighbor_names = get_neighbors(src_ID,relations,reversed_relations)
    _, _, tgt_neighbor_names, tgt_reversed_neighbor_names = get_neighbors(tgt_ID,relations,reversed_relations)
    # return concept_data[tgt_ID],tgt_ID,tgt_neighbor_names
    src_concept = concept_data[src_ID]
    tgt_concept = concept_data[tgt_ID]
    in_context = ""
    if len(src_reversed_neighbor_names)>0:
        in_context+="We know that "+ src_concept + " is a prerequisite of the following concepts: "+ ','.join(src_reversed_neighbor_names) +";"
    if len(src_neighbor_names) > 0 :
        in_context+=" The following concepts are the prerequisites of "+src_concept+" : "+','.join(src_neighbor_names)+';'
    if len(tgt_reversed_neighbor_names)>0:
        in_context+="We know that "+ tgt_concept + " is a prerequisite of the following concepts: "+ ','.join(tgt_reversed_neighbor_names) +";"
    if len(tgt_neighbor_names) > 0 :
        in_context+=" The following concepts are the prerequisites of "+tgt_concept+" : "+','.join(tgt_neighbor_names)+';'
    # print (in_context)

    prompt = template_head + src_concept + ' , ' + tgt_concept + template_mid + in_context + template_tail
    return prompt

    
if __name__ == "__main__":
    
    # batch_id = '0'
    # for batch_id in ['1','2','3','4']:
    # for batch_id in ['0']:
    for batch_id in ['0', '1', '2', '3', '4']:
        relations,reversed_relations=load_train(batch_id)

        label_path_pos = data_path + annotation_positive + batch_id + '.txt'
        label_path_neg = data_path + annotation_negative + batch_id + '.txt'
        os.makedirs(instruction_path, exist_ok=True)
        with open(instruction_path+'t2.pos.'+batch_id+'.txt','w') as writer:
            with open(label_path_pos,'r') as file:
                for line in file:
                    source, target=line.strip().split(',')
                    writer.write(get_content(int(source),int(target),relations,reversed_relations))
        with open(instruction_path+'t2.neg.'+batch_id+'.txt','w') as writer:
            with open(label_path_neg,'r') as file:
                for line in file:
                    source, target=line.strip().split(',')
                    writer.write(get_content(int(source),int(target),relations,reversed_relations))


                    # import pdb;pdb.set_trace()
                    # pass
    

