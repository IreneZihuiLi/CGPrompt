rag_folder_path = ''
from embedchain import App
import os
os.environ["OPENAI_API_KEY"] = "sk-"

config_dict = {
  'llm': {
    'provider': 'openai',
    'config': {
      'model': 'gpt-3.5-turbo',
      'temperature': 0.0,
      'max_tokens': 20,
      'top_p': 1,
      'stream': False
    }
  },
  'embedder': {
    'provider': 'openai'
  }
}


# load llm configuration from config dict
app = App.from_config(config=config_dict)
app.add(rag_folder_path, data_type="directory")

print ('Build RAG finished.')

data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
annotation_positive = 'split/test_edges_positive_'
annotation_negative = 'split/test_edges_negative_'
RAG_res_path = '../RAG_res/0123/GPT3/'

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


prompt_test = """We have two {domain} related concepts: A: "{concept_1}" and B: "{concept_2}".
Do you think that people learn "{concept_1}" will help understand "{concept_2}"?

Hint:
1. Answer YES or NO only.
2. This is a directional relation, which means if YES, (B,A) may be False, but (A,B) is True.
3. Your answer will be used to create a knowledge graph.
"""
domain = "Natural Language Processing"  # Replace with your domain


def generate_answer(batch_id):
  batch_id = str(batch_id)
  label_path_neg = data_path + annotation_negative + batch_id + '.txt'
  label_path_pos = data_path + annotation_positive + batch_id + '.txt'

  os.makedirs(RAG_res_path, exist_ok=True)
  with open(RAG_res_path + 't1.neg.' + batch_id + '.txt', 'w') as writer:
    with open(label_path_neg, 'r') as file:
      for line in file:
        a, b = line.strip().split(',')
        source = concept_data[int(a)]
        target = concept_data[int(b)]
        query_str = prompt_test.format(domain=domain, concept_1=source, concept_2=target)
        res = app.query(query_str)
        writer.write(res + '\n')
        # import pdb;pdb.set_trace()
        # pass

  with open(RAG_res_path + 't1.pos.' + batch_id + '.txt', 'w') as writer:
    with open(label_path_pos, 'r') as file:
      for line in file:
        a, b = line.strip().split(',')
        source = concept_data[int(a)]
        target = concept_data[int(b)]
        query_str = prompt_test.format(domain=domain, concept_1=source, concept_2=target)
        res = app.query(query_str)
        writer.write(res+'\n')

# generate_answer(0)
#
for i in range(3,5):
    generate_answer(i)
    print ('Finished ',str(i))

# run pos 2
batch_id = '2'
label_path_neg = data_path + annotation_negative + batch_id + '.txt'
label_path_pos = data_path + annotation_positive + batch_id + '.txt'

print ('Now pos 2 ',str(i))
with open(RAG_res_path + 't1.pos.' + batch_id + '.txt', 'w') as writer:
  with open(label_path_pos, 'r') as file:
    for line in file:
      a, b = line.strip().split(',')
      source = concept_data[int(a)]
      target = concept_data[int(b)]
      query_str = prompt_test.format(domain=domain, concept_1=source, concept_2=target)
      res = app.query(query_str)
      writer.write(res+'\n')

