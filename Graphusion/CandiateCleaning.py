# -*- coding: utf-8 -*-


# !pip install --upgrade openai
from google.colab import drive
drive.mount('/content/drive')

file_path = ''
import json,re

res_name="gpt-4-turbo"
# res_name= 'gpt-4o'
# res_name='gpt-3.5-turbo'


relation_list = ['Compare','Part-of','Conjunction','Evaluate-for','Is-a-Prerequisite-of','Used-for','Hyponym-Of']

with open(file_path+res_name+'.txt') as f:
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

for triplet in filtered_triplets[:40]:
    print(triplet)
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
with open(file_path+res_name+'-stats.txt','a') as w:
  for relation, count in relation_count.items():
    print(f"{relation}: {count}")
    w.write(f"{relation}: {count}\n")

with open(file_path+res_name+'-triplets.txt','a') as w:
    for item in filtered_triplets:
        w.write(str(item)+'\n')