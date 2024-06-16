'''
calculate gpt results
NOTE: MAP and AUC are wrong
'''

# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score



batch = '0'
# res_path_neg='../results/1113_gpt/t1.neg.'+batch+'.txt.test'
# res_path_pos='../results/1113_gpt/t1.pos.'+batch+'.txt.test'

# res_path_neg='../results/1206_train_wiki/t1.neg.'+batch+'.txt.test'
# res_path_pos='../results/1206_train_wiki/t1.pos.'+batch+'.txt.test'
res_path_neg='../results/1205_train/t2.neg.'+batch+'.txt.test'
res_path_pos='../results/1205_train/t2.pos.'+batch+'.txt.test'


pred = []
truth = []

def load_res(res_path):
    if 'neg' in res_path:
        flag = 0
    else:
        flag = 1
    with open(res_path,'r') as f:
        for line in f.readlines():
            line = line.strip().lower()
            if line.startswith('no') or line.startswith('the answer is no'):
                pred.append(0)
            else:
                pred.append(1)
            truth.append(flag)
    return pred, truth

y_1,t_1=load_res(res_path_neg)
y_2,t_2=load_res(res_path_pos)
y_test = t_1+t_2
y_pred = y_1+y_2


# Calculate various metrics
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
map_score = average_precision_score(y_test, y_pred)  # Using probabilities for MAP
auc = roc_auc_score(y_test, y_pred)  # Using probabilities for AUC

# Output the results
# print(f"Accuracy: {accuracy:.4f}")
# print(f"F1 Score: {f1:.4f}")
# print(f"Mean Average Precision (MAP): {map_score:.4f}")
# print(f"Area Under Curve (AUC): {auc:.4f}")

print (f"Acc, F1: {accuracy:.4f},{f1:.4f}")


# import pdb;pdb.set_trace()
# Calculating precision, recall, and F1 score
# precision = precision_score(y_true, y_pred)
# recall = recall_score(y_true, y_pred)
# f1 = f1_score(y_true, y_pred)
#
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1 Score:", f1)


'''
GPT4: 0
Precision: 0.6045197740112994
Recall: 0.6903225806451613
F1 Score: 0.6445783132530121

Accuracy: 0.6194
F1 Score: 0.6446
Mean Average Precision (MAP): 0.5722
Area Under Curve (AUC): 0.6194

0.6258,0.7157


1
Precision: 0.8775510204081632
Recall: 0.5548387096774193
F1 Score: 0.6798418972332015

Accuracy: 0.7387
F1 Score: 0.6798
Mean Average Precision (MAP): 0.7095
Area Under Curve (AUC): 0.7387

-----
GPT 3.5 0
Accuracy: 0.6806
F1 Score: 0.7114

Accuracy: 0.7273
F1 Score: 0.7837






'''
