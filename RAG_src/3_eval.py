from sklearn.metrics import accuracy_score, f1_score, average_precision_score, roc_auc_score

# RAG_res_path = '../RAG_res/0123/GPT4/'
RAG_res_path = '../RAG_res/0124/GPT4/'

def get_res(batch='0'):

    res_path_neg=RAG_res_path+'t1.neg.'+batch+'.txt'
    res_path_pos=RAG_res_path+'t1.pos.'+batch+'.txt'


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
    return accuracy,f1


acc_list, f1_list = [], []
for batch in range(5):
    acc, f1 = get_res(str(batch))
    acc_list.append(acc)
    f1_list.append(f1)
print ('Final..\n')
print(sum(acc_list)/5.,sum(f1_list)/5.)


'''
GPT3 RAG 30
Acc, F1: 0.7581,0.7761
Acc, F1: 0.7742,0.7879


GPT3 RAG 25
Acc, F1: 0.7710,0.7829
Acc, F1: 0.7903,0.8000


'''