import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import random

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset



## ------- Defination -----------
# feature_file = './res/concept_emb_openai_3152.pkl'
feature_file = './res/concept_emb_llama70b_8192.pkl'
data_path = '../concept_data/'
seed = 42

class EdgeDataset(Dataset):
    def __init__(self, positive_data, negative_data):
        self.data = positive_data + negative_data
        random.shuffle(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        edge, label = self.data[idx]
        return edge, float(label)

def load_dataset(data_path, train=True):
    dataset_pos = []
    dataset_neg = []
    if train:
        positive_files = [data_path + f'split/train_edges_positive_{i}.txt' for i in range(5)]
        negative_files = [data_path + f'split/train_edges_negative_{i}.txt' for i in range(5)]
    else:
        positive_files = [data_path + f'split/test_edges_positive_{i}.txt' for i in range(5)]
        negative_files = [data_path + f'split/test_edges_negative_{i}.txt' for i in range(5)]

    for file in positive_files:
        with open(file, 'r') as f:
            for line in f:
                edge = tuple(map(lambda x: int(x), line.strip().split(',')))
                dataset_pos.append((edge, 1))

    for file in negative_files:
        with open(file, 'r') as f:
            for line in f:
                edge = tuple(map(lambda x: int(x), line.strip().split(',')))
                dataset_neg.append((edge, 0))

    if train:
        random.shuffle(dataset_neg)
        dataset_neg = dataset_neg[:len(dataset_pos)]
        return EdgeDataset(dataset_pos, dataset_neg)
    else:
        return EdgeDataset(dataset_pos, dataset_neg)

def build_graph(data_path):
    edge_list = []
    for batch_id in range(5):
        for prefix in ['split/train_edges_positive_', 'split/train_edges_negative_']:
            file_path = data_path + prefix + str(batch_id) + '.txt'
            with open(file_path, 'r') as file:
                for line in file:
                    edge = [int(x) for x in line.strip().split(',')]
                    edge_list.append(edge[:2])

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    return edge_index

def load_node_features(feature_file, tensor=True):
    with open(feature_file, 'rb') as f:
        concept_generated_emb = pickle.load(f)

    if tensor==True:
        return torch.stack(concept_generated_emb).float()
    else:
        return torch.from_numpy(np.array(concept_generated_emb)).float()

class GCN(torch.nn.Module):
    def __init__(self, num_features, hidden_channels):
        super(GCN, self).__init__()

        self.linear = nn.Linear(num_features, 256)
        self.conv1 = GATConv(256, 256)
        self.conv2 = GATConv(256, 128)

    def forward(self, x, edge_index, edge):
        x = self.linear(x)
        x = F.relu(x)

        # GCN layers
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)

        edge_features = (x[edge[0]] * x[edge[1]]).sum(dim=1)

        return torch.sigmoid(edge_features)


if __name__ == "__main__":
    torch_geometric.seed_everything(seed)

    edge_index = build_graph(data_path)
    x = load_node_features(feature_file)

    data = Data(x=x, edge_index=edge_index).to('cuda:0')

    model = GCN(num_features=x.size(1), hidden_channels=64).to('cuda:0')
    criterion = torch.nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    train_data = load_dataset(data_path, train=True)
    test_data = load_dataset(data_path, train=False)
    batch_size = 32
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)


    def train():
        model.train()
        total_loss = 0
        for idx, (edge, label) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            label = label.float().to("cuda:0")

            output = model(data.x, data.edge_index, edge)
            # import pdb;pdb.set_trace()
            loss = criterion(output.squeeze(-1), label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        return total_loss / len(train_data)

    def evaluate(model, data, test_data):
        model.eval()
        y_true = []
        y_pred = []

        with torch.no_grad():
            for edge, label in test_loader:
                output = model(data.x, data.edge_index, edge)
                prediction = (output > 0.5).float()
                y_true.extend(label.cpu().tolist())
                y_pred.extend(prediction.cpu().tolist())

        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        return acc, f1

    num_epochs = 100
    for epoch in range(num_epochs):
        print(f"epoch: {epoch}")
        loss = train()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss:.7f}')

        accuracy, f1 = evaluate(model, data, test_data)
        print(f'Test Accuracy: {accuracy:.7f}, F1 Score: {f1:.7f}')