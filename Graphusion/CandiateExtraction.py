# -*- coding: utf-8 -*-

# !pip install --upgrade openai

from google.colab import drive
drive.mount('/content/drive')


file_path = 'concept_abstracts_70.json'
import json
with open(file_path) as f:
    data = json.load(f)

concept_list = [x for x in data.keys()]
print (len(concept_list))
print (concept_list[:5])

keyword='gated recurrent unit'
single_abs = data[keyword]['abstracts']
single_abs = ' '.join(single_abs)
single_abs

import os
os.environ['OPENAI_API_KEY'] = 'sk-'
from openai import OpenAI
client = OpenAI()

instruction = '''
### Instruction:
You are a domain expert in computer science, natural language processing, and now you are building a knowledge graph in this domain. Given a context (### Content), and a query concept (### Concept), do the following:
1. Extract the query concept and some in-domain concepts from the context, these concepts should be fine-grained: could be introduced by a lecture slide page, or a whole lecture, or possibly to have a Wikipedia page.
2. Determine the relationships between the query concept and the extracted concepts from Step 1, in a triplet format: (<head concept>, <relation>, <tail concept>). The relationship should be functional, aiding learners in understanding the knowledge. The query concept can be the head concept or tail concept. We define 7 types of the relations:
    a) Compare: Represents a relationship between two or more entities where a comparison is being made. For example, "A is larger than B" or "X is more efficient than Y."
    b) Part-of: Denotes a relationship where one entity is a constituent or component of another. For instance, "Wheel is a part of a Car."
    c) Conjunction: Indicates a logical or semantic relationship where two or more entities are connected to form a group or composite idea. For example, "Salt and Pepper."
    d) Evaluate-for: Represents an evaluative relationship where one entity is assessed in the context of another. For example, "A tool is evaluated for its effectiveness."
    e) Is-a-Prerequisite-of: This dual-purpose relationship implies that one entity is either a characteristic of another or a required precursor for another. For instance, "The ability to code is a prerequisite of software development."
    f) Used-for: Denotes a functional relationship where one entity is utilized in accomplishing or facilitating the other. For example, "A hammer is used for driving nails."
    g) Hyponym-Of: Establishes a hierarchical relationship where one entity is a more specific version or subtype of another. For instance, "A Sedan is a hyponym of a Car."
3. Some relation types are strictly directional. For example, "A tool is evaluated for B" indicates (A, Evaluate-for, B), NOT (B, Evaluate-for, A). Among the seven relation types, only "a) Compare" and "c) Conjunction" are not direction-sensitive.
4. You can also extract triplets from the extracted concepts, and the query concept may not be necessary in the triplets.
5. Your answer should ONLY contain a list of triplets, each triplet is in this format: (concept, relation, concept). For example: "(concept, relation, concept)(concept, relation, concept)." No numbering and other explanations are needed.
6. If ### Content is empty, output None.


### Content:
'''

res_path = '../'+model_name+'.txt'
for id,keyword in enumerate(concept_list):
  # following is the test code
  instruction2 = '''\n\n### Concept:'''+keyword

  with open(res_path,'a') as w:
      abstracts = data[keyword]['abstracts']
      abstracts = ' '.join(abstracts)[:14500]
      ft = open(res_path+'ft.txt','w')
      # print (instruction + abstracts + instruction2)
      try:
        # Get the answer from GPT
        answer = get_answer_GPT(instruction + abstracts + instruction2)
      except Exception as e:
          print("error")
          print (e)
          answer = "error"

      # answer = get_answer_GPT(instruction + abstracts + instruction2)
      print(str(id)+' '+concept_list[id]+':\n'+answer.replace('\n',' ')+'\n\n')
      w.write(answer.replace('\n',' ')+'\n')
      w.flush()

model_name="gpt-4-turbo"
# model_name='gpt-4o'
# model_name='gpt-3.5-turbo'

def get_answer_GPT(instruction):
    completion = client.chat.completions.create(
      model=model_name,
      messages=[
        {"role": "system", "content": "You are a knowledge graph builder."},
        {"role": "user", "content": instruction}
      ],
      max_tokens=200  # Sets the maximum length of the response
    )

    # Ensure that the response is correctly accessed
    if completion.choices:
        # print(completion.choices[0].message.content)
        answer = completion.choices[0].message.content.replace('\n','')
    else:
        print("No completion found.")
        answer = 'ERROR'

    return answer