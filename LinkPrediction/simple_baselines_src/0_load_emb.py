import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import random
from sklearn.svm import SVC
import statistics as stat
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier




def get_res(batch_id='0'):
    n_components = 200

    emb_path = ''
    data_path = '../concept_data/'
    concept_path = data_path + '322topics_final.tsv'
    annotation_positive = 'split/train_edges_positive_'
    annotation_negative = 'split/train_edges_negative_'
    label_path_neg = data_path + annotation_negative + batch_id + '.txt'
    label_path_pos = data_path + annotation_positive + batch_id + '.txt'
    test_positive = 'split/test_edges_positive_'
    test_negative = 'split/test_edges_negative_'
    test_path_neg = data_path + test_negative + batch_id + '.txt'
    test_path_pos = data_path + test_positive + batch_id + '.txt'



        
    # Open the file in binary read mode
    with open(emb_path, 'rb') as file:
        # Load the data from the file
        data = pickle.load(file)

    # Now 'data' contains the contents of your pickle file
    print(len(data)) # (322,8192) list



    concept_data = dict()
    # load concept as dict
    with open(concept_path, 'r') as file:
        for line in file:
            # Split each line at the pipe character
            key, value = line.strip().split('|')
            # Convert key to an integer and strip any whitespace from the value
            concept_data[int(key)-1] = value.strip()

    print (len(concept_data)," concepts loaded.")




    # def load_features(file_name, pos=True):
    #     x = []
    #     y = []

    #     with open(file_name,'r') as file:
    #         for line in file:
    #             source, target=line.strip().split(',')
    #             source_emb = data[int(source)]
    #             target_emb = data[int(target)]
    #             feature = np.concatenate((source_emb, target_emb))
    #             x.append(feature)
    #             if pos:
    #                 y.append(1)
    #             else:
    #                 y.append(0)
    #     return x, y


    train_x = []
    train_y= []


    with open(label_path_pos,'r') as file:
        for line in file:
            source, target=line.strip().split(',')
            source_emb = data[int(source)]
            target_emb = data[int(target)]
            
            feature = np.concatenate((source_emb.cpu().numpy(), target_emb.cpu().numpy()))
            train_x.append(feature)
            train_y.append(1)

    # random sample on negative training data
    train_x_neg = []
    with open(label_path_neg,'r') as file:
        for line in file:
            source, target=line.strip().split(',')
            source_emb = data[int(source)]
            target_emb = data[int(target)]
            feature = np.concatenate((source_emb.cpu().numpy(), target_emb.cpu().numpy()))
            train_x_neg.append(feature)

    # sampling and making x, y for training
    len_train_pos = len(train_x)
    for i in range(len_train_pos): train_y.append(0)
    train_x_neg_sampled = random.sample(train_x_neg, len_train_pos)
    train_x = np.concatenate((train_x,train_x_neg_sampled))

    # import pdb;pdb.set_trace()
    # start training
    train_y = np.array(train_y)
    train_x = np.array(train_x)

    print ('Shape:',train_x.shape,train_y.shape)

    # # Create a TruncatedSVD instance
    svd = TruncatedSVD(n_components=n_components)

    train_x = svd.fit_transform(train_x)

    print ('Start PCA')
    # Optional: Initial dimensionality reduction using PCA
    # train_x = PCA(n_components=n_components).fit_transform(train_x)  # Reduce to 50 dimensions

    print ('Start training')
    # Create a logistic regression model
    # model = LogisticRegression()
    # model = SVC()
    # model = GaussianNB()
    # model = RandomForestClassifier(n_estimators=100, random_state=42)
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)



    # Train the model
    model.fit(train_x, train_y)


    test_x = []
    test_y = []

    with open(test_path_pos,'r') as file:
        for line in file:
            source, target=line.strip().split(',')
            source_emb = data[int(source)]
            target_emb = data[int(target)]
            feature = np.concatenate((source_emb.cpu().numpy(), target_emb.cpu().numpy()))
            test_x.append(feature)
            test_y.append(1)

    with open(test_path_neg,'r') as file:
        for line in file:
            source, target=line.strip().split(',')
            source_emb = data[int(source)]
            target_emb = data[int(target)]
            feature = np.concatenate((source_emb.cpu().numpy(), target_emb.cpu().numpy()))
            test_x.append(feature)
            test_y.append(0)
    # import pdb;pdb.set_trace()

    print ('Start evaluation')
    # Make predictions on the test set
    test_x = svd.fit_transform(test_x)
    # test_x = PCA(n_components=n_components).fit_transform(test_x)
    predictions = model.predict(test_x)



    # Calculate accuracy
    accuracy = accuracy_score(test_y, predictions)

    # Calculate F1 score
    # Note: You might need to specify the 'average' parameter depending on your classification type
    # For binary classification, the default 'binary' is usually fine
    f1 = f1_score(test_y, predictions, average='binary')

    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")
    
    return accuracy, f1
   

if __name__ == "__main__":
    accuracy = []
    F1 = []
    
    for i in range(5):
        acc,f1 = get_res(str(i))
        accuracy.append(acc)
        F1.append(f1)
    print ('Done..')
    print ('Acc, F1:',stat.mean(accuracy), stat.mean(F1))

# import pdb;pdb.set_trace()
# pass


'''
Accuracy: 0.5129032258064516
F1 Score: 0.5519287833827893


PCA 100
Acc, F1: 0.5612903225806452 0.4924134094187201

150
Acc, F1: 0.5819354838709677 0.48463814446715087

200
0.5496774193548387 0.4149615656532305


NB！
SVD 100
Acc, F1: 0.6561290322580645 0.6181233298342588

150
Acc, F1: 0.6645161290322581 0.6418079901424246

SVD 200
Acc, F1: 0.6683870967741935 0.6474817964898676

250
Acc, F1: 0.64 0.6261834972227311

SVD 300
0.6432258064516129 0.6635690060697268


--- SVD, random forest, classifier = 100
SVD 250
0.5974193548387097 0.6097205052901027

SVD 100
Acc, F1: 0.6219354838709678 0.6225234013919787

150
Acc, F1: 0.6296774193548387 0.6267434602394328

SVD 200
Acc, F1: 0.6141935483870968 0.6144746468326631

300
Acc, F1: 0.603225806451613 0.634785058222437


GBC
100
Acc, F1: 0.6154838709677419 0.6108665915565653

200
Acc, F1: 0.6070967741935483 0.6156784764690529

'''
