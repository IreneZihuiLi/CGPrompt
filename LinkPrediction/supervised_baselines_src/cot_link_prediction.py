import urllib
from openai import OpenAI
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score
import re
import csv
from sklearn.metrics import confusion_matrix

## ------------ files --------------------
result_path = './res/cot_test_dataset_msg.csv'
data_path = '../concept_data/'
concept_path = data_path + '322topics_final.tsv'
annotation = data_path + 'final_new_annotation.csv'
model_name = "gpt-4-1106-preview"
msg_path = result_path

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

DOMAIN = """NLP"""

SYSTEM_MESSAGE = """You are a Knowledge Graph Expert. Your task is to predict the relations between nodes in a knowledge graph using the Chain of Thought method. Explain your reasoning process step by step and conclude with a clear YES or NO answer, marked distinctly."""

LINK_PREDICTION_MESSAGE = """In the context of {domain}, we have two concepts: A: '{concept_1}' and B: '{concept_2}'. Assess if understanding '{concept_1}' is a necessary prerequisite for understanding '{concept_2}'. Employ the Chain of Thought approach to detail your reasoning before giving a final answer.

# Identify the Domain and Concepts: Clearly define A and B within their domain. Understand the specific content and scope of each concept.

# Analyze the Directional Relationship: Determine if knowledge of concept A is essential before one can fully grasp concept B. This involves considering if A provides foundational knowledge or skills required for understanding B.

# Evaluate Dependency: Assess whether B is dependent on A in such a way that without understanding A, one cannot understand B.

# Draw a Conclusion: Based on your analysis, decide if understanding A is a necessary prerequisite for understanding B.

# Provide a Clear Answer: After detailed reasoning, conclude with a distinct answer: '<result>YES</result>' if understanding A is a prerequisite for understanding B, or '<result>NO</result>' if it is not.
"""

if __name__ == '__main__':

    concept_data = dict()
    with open(concept_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('|')
            concept_data[int(key)-1] = value.strip()
    print (len(concept_data)," concepts loaded.")

    # read every pair
    test_pair_list = []
    for file in test_path_list:
        with open(file, 'r') as f:
            if 'positive' in file:
                ground_truth = 1
            else:
                ground_truth = 0

            lines = f.readlines()
            for l in lines:
                concept_id1, concept_id2 = l.strip().split(',')
                concept_name1 = concept_data[int(concept_id1)]
                concept_name2 = concept_data[int(concept_id2)]
                test_pair_list.append({
                    "c1": concept_name1,
                    "c2": concept_name2,
                    "truth": ground_truth,
                })

    # ask openai
    print("Open AI Start")
    client = OpenAI()
    results_json = []

    for dict in test_pair_list:
        concept_name1, concept_name2, label = dict['c1'], dict['c2'], dict['truth']

        msg = [
                {"role": "system",
                    "content": SYSTEM_MESSAGE},
                {"role": "user", "content": LINK_PREDICTION_MESSAGE.format(
                domain=DOMAIN, concept_1=concept_name1, concept_2=concept_name2
            )}
            ]
        completion = client.chat.completions.create(
            model=model_name,
            messages= msg,
            temperature = 0,
        )

        res = completion.choices[0].message.content
        print("Msg: ", msg)
        print("Resp: ", res)
        print("Truth: ", label)
        print("=================================")
        results_json.append({
            "msg": msg,
            "ground_truth": label,
            "resp": res
        })
    json.dump(results_json, open(result_path, 'w'), indent=4)

    y_pred = []
    y_test = []
    error = 0

    res = json.load(open(result_path, 'r'))
    for r in res :
        if re.search('<result>YES</result>', r['resp'], re.IGNORECASE):
        # if re.search('YES', r['resp'], re.IGNORECASE):
            y_pred.append(1)
        elif re.search('<result>NO</result>', r['resp'], re.IGNORECASE):
        # elif re.search('NO', r['resp'], re.IGNORECASE):
            y_pred.append(0)
        else:
            error += 1
            print(r['resp'])
            continue

        if r['ground_truth'] == 1:
            y_test.append(1)
        else:
            y_test.append(0)

        # result_pair.append([concept_id1, concept_id2])

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