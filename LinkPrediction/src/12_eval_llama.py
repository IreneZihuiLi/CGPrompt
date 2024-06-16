'''
calculate gpt results
NOTE: MAP and AUC are wrong
'''
import sys

import numpy as np
# from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score


def get_score(batch='0', res_path_neg='', res_path_pos=''):
    def load_res(res_path):
        pred = []
        truth = []

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

    # import pdb;pdb.set_trace()
    y_test = t_1+t_2
    y_pred = y_1+y_2

    print('neg')
    print(np.unique(y_1, return_counts=True))
    print('pos')
    print(np.unique(y_2, return_counts=True))

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

    # print(f"ACC, F1: {accuracy:.4f},{f1:.4f}")
    print(f"{accuracy:.4f},{f1:.4f}")
    return accuracy, f1

if __name__ == "__main__":
# # <<<<<<< HEAD
#     # """ this is for NLP data, modify by sixun, uncomment to run """
#     acc_list, f1_list = [], []
#     for batch in range(5):
#         batch = str(batch)
#         res_path_neg = '../results/1206_train_wiki/t1.neg.' + batch + '.txt.test'
#         res_path_pos = '../results/1206_train_wiki/t1.pos.' + batch + '.txt.test'
#         accuracy, f1 = get_score(str(batch), res_path_neg=res_path_neg, res_path_pos=res_path_pos)
#         acc_list.append(accuracy)
#         f1_list.append(f1)
#     print('avg accuracy {}, avg f1 {}'.format(np.mean(acc_list), np.mean(f1_list)))

    # NLP trained by paper
    acc_list, f1_list = [], []
    for batch in range(5):
        batch = str(batch)
        res_path_neg = '../results/NLP_paper/t1.neg.' + batch + '.txt.test'
        res_path_pos = '../results/NLP_paper/t1.pos.' + batch + '.txt.test'
        accuracy, f1 = get_score(str(batch), res_path_neg=res_path_neg, res_path_pos=res_path_pos)
        acc_list.append(accuracy)
        f1_list.append(f1)
    print('avg accuracy {}, avg f1 {}'.format(np.mean(acc_list), np.mean(f1_list)))

    # """ this is for CV/BIO, uncomment to run """
    # domain = 'BIO'
# =======
    # # """ this is for NLP data, modify by sixun, uncomment to run """
# >>>>>>> 1a772b929ebde7f700c31029dc81cc59db3f1a8d
    # acc_list, f1_list = [], []
    # for batch in range(5):
    #     batch = str(batch)
    #     res_path_neg = '../results/1205_train/t3.neg.' + batch + '.txt.test'
    #     res_path_pos = '../results/1205_train/t3.pos.' + batch + '.txt.test'
    #     accuracy, f1 = get_score(str(batch), res_path_neg=res_path_neg, res_path_pos=res_path_pos)
    #     acc_list.append(accuracy)
    #     f1_list.append(f1)
    # print('avg accuracy {}, avg f1 {}'.format(np.mean(acc_list), np.mean(f1_list)))

    # """ this is for CV/BIO, uncomment to run """
    # domain = 'BIO_train'
    # acc_list, f1_list = [], []
    # for batch in range(5):
    #     batch = str(batch)
    #     res_path_neg = '../results/{}/t2.neg.{}.txt.test'.format(domain, batch)
    #     res_path_pos = '../results/{}/t2.pos.{}.txt.test'.format(domain, batch)
    #     accuracy, f1 = get_score(str(batch), res_path_neg=res_path_neg, res_path_pos=res_path_pos)

    #     acc_list.append(accuracy)
    #     f1_list.append(f1)

    # print('avg accuracy {}, avg f1 {}'.format(np.mean(acc_list), np.mean(f1_list)))

    # """ this is for CV/BIO - GPT, uncomment to run """
    # domain = 'BIO' #'CV'
    # acc_list, f1_list = [], []
    # for batch in range(5):
    #     batch = str(batch)
    #     res_path_neg = '../results/{}_gpt35/t1.neg.{}.txt.test'.format(domain, batch)
    #     res_path_pos = '../results/{}_gpt35/t1.pos.{}.txt.test'.format(domain, batch)
    #     accuracy, f1 = get_score(str(batch), res_path_neg=res_path_neg, res_path_pos=res_path_pos)

    #     acc_list.append(accuracy)
    #     f1_list.append(f1)

    # print('avg accuracy {}, avg f1 {}'.format(np.mean(acc_list), np.mean(f1_list)))

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
LLAMA2-70b batch 0
Accuracy: 0.6097
F1 Score: 0.6921
Mean Average Precision (MAP): 0.5627
Area Under Curve (AUC): 0.6097

1





'''