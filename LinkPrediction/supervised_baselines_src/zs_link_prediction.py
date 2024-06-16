import urllib
from openai import OpenAI
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re
import csv
from sklearn.metrics import confusion_matrix

## --------- Files Path -------------------
final_pairs_path = './generated_pair.csv'
result_path = './zs_test_dataset_msg.csv'
data_path = '../concept_data/'
model_name = 'gpt-4-1106-preview'
concept_path = data_path + '322topics_final.tsv'
annotation = data_path + 'final_new_annotation.csv'
test_path_list = [
    data_path + 'split/test_edges_positive_0.txt',
    data_path + 'split/test_edges_negative_0.txt',
    data_path + 'split/test_edges_positive_1.txt',
    data_path + 'split/test_edges_negative_1.txt',
    data_path + 'split/test_edges_positive_2.txt',
    data_path + 'split/test_edges_negative_2.txt',
    data_path + 'split/test_edges_positive_3.txt',
    data_path + 'split/test_edges_negative_3.txt',
    data_path + 'split/test_edges_positive_4.txt',
    data_path + 'split/test_edges_negative_4.txt',
]
## --------- Defiantion -------------------

DOMAIN = """NLP"""

SYSTEM_MESSAGE = """You are a Knowledge Graph Expert.
You are going to build a knowledge graph or predict the relations between nodes following the instructions below:

1. Solve the task step by step if you need to.
2. When you find an answer, verify the answer carefully.
3. You must follow the user's instructions to answer the question.
4. Do not think that anything is connected. There is a high chance that some concepts are not related.
5. Some questions may be asked of you multiple times, and you can doubt your previous answer and provide a new one.
"""

LINK_PREDICTION_MESSAGE = """We have two {domain} related concepts: A: "{concept_1}" and B: "{concept_2}".
Do you think that people learn "{concept_1}" will help understand "{concept_2}"?

Hint:
1. Answer YES or NO only.
2. This is a directional relation, which means if YES, (B,A) may be False, but (A,B) is True.
3. Your answer will be used to create a knowledge graph.
"""

if __name__ == '__main__':
    concept_data = dict()
    with open(concept_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|')
            concept_data[int(key)-1] = value.strip()
    print (len(concept_data)," concepts loaded.")



    client = OpenAI()
    results = []
    with open(result_path,'w') as w:
        with open(final_pairs_path,'r') as f:
            for line in f:
                concept_id1, concept_id2, label = line.strip().split(',')
                concept_name1 = concept_data[int(concept_id1)]
                concept_name2 = concept_data[int(concept_id2)]

                msg = [
                        {"role": "system",
                            "content": SYSTEM_MESSAGE},
                        {"role": "user", "content": LINK_PREDICTION_MESSAGE.format(
                        domain=DOMAIN, concept_1=concept_name1, concept_2=concept_name2
                    )}
                    ]
                completion = client.chat.completions.create(
                    # model="gpt-3.5-turbo",
                    model="gpt-4-1106-preview",
                    messages= msg,
                    temperature = 0,
                    max_tokens = 100
                )

                res = completion.choices[0].message.content
                print("Msg: ", msg)
                print("Resp: ", res)
                print("Truth: ", label)
                print("=================================")
                results.append({
                    "msg": msg,
                    "ground_truth": label,
                    "resp": res
                })
        json.dump(results, w, indent=4)

    y_pred = []
    y_test = []
    result_pair = []
    error = 0
    csv_filename = './result_pair.csv'
    with open(result_path, 'r') as f:
        res = json.load(f)
        with open(final_pairs_path,'r') as f:
            with open(csv_filename, 'w', newline='') as file:
                writer = csv.writer(file)

                for line, r in zip(f, res):
                    concept_id1, concept_id2, label = line.strip().split(',')

                    if re.search('yes', r['resp'], re.IGNORECASE) and not re.search('no', r['resp'], re.IGNORECASE):
                        y_pred.append(1)
                        writer.writerow([concept_id1, concept_id2])
                    elif re.search('no', r['resp'], re.IGNORECASE) and not re.search('yes', r['resp'], re.IGNORECASE):
                        y_pred.append(0)
                    else:
                        error += 1
                        print(r['resp'])
                        continue

                    if r['ground_truth'] == '1':
                        y_test.append(1)
                    else:
                        y_test.append(0)

                    result_pair.append([concept_id1, concept_id2])

    print(f"error count: {error}")
    print(f"Final accuracy: {accuracy_score(y_test, y_pred)}")
    print(f"Final f1: {f1_score(y_test, y_pred)}")

    # 计算混淆矩阵
    confusion = confusion_matrix(y_test, y_pred)

    # 提取四个象限的数量
    tp = confusion[1][1]  # True Positive
    tn = confusion[0][0]  # True Negative
    fp = confusion[0][1]  # False Positive
    fn = confusion[1][0]  # False Negative

    print(f"True Positive (TP): {tp}")
    print(f"True Negative (TN): {tn}")
    print(f"False Positive (FP): {fp}")
    print(f"False Negative (FN): {fn}")