from MyNets import MyDataset
import random
import torch
import torch.nn.functional as F

from torch_geometric.data import DataLoader, Data
from sklearn import metrics
from sklearn.metrics import roc_curve, auc, roc_auc_score, accuracy_score
from sklearn.metrics import classification_report
from Deal_Kmer import *
from dataloader import dataload
from model import RNA2vec_KNFP_GCN_2layer
import pandas as pd

proteins = ['WTAP', 'AGO1', 'AGO2', 'AGO3', 'ALKBH5', 'AUF1', 'C17ORF85', 'C22ORF28', 'CAPRIN1',
            'DGCR8', 'EIF4A3', 'EWSR1', 'FMRP', 'FOX2', 'FUS', 'FXR1', 'FXR2', 'HNRNPC', 'HUR', 'IGF2BP1', 'IGF2BP2',
            'IGF2BP3', 'LIN28A', 'LIN28B', 'METTL3', 'MOV10', 'PTB', 'PUM2', 'QKI', 'SFRS1', 'TAF15', 'TDP43',
            'TIA1', 'TIAL1', 'TNRC6', 'U2AF65', 'WTAP', 'ZC3H7B']

batch_size = 50
for protein in proteins:
    all_graphs = dataload(protein)
    # print(all_graphs[0].is_undirected())

    random.seed(1)
    random.shuffle(all_graphs)

    n = len(all_graphs) // 5
    test_dataset = MyDataset(all_graphs[4 * n:])  # 必须先只取后面的值
    train_graphs = all_graphs[:4 * n]
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    labels = [g.y for g in train_graphs]
    device = torch.device("cuda:1")


    def train(loader, data_len):
        model.train()
        loss_all = 0
        for data in loader:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, data.y)
            loss.backward()
            loss_all += data.num_graphs * loss.item()
            optimizer.step()
        return loss_all / data_len


    def test(loader):
        model.eval()
        score_all = []
        test_labels = []
        y_pre = []
        correct = 0
        for data in loader:
            data = data.to(device)
            pred = model(data).max(dim=1)[1]
            y_pre.extend(pred.tolist())
            correct += pred.eq(data.y).sum().item()
            output = model(data)
            output = torch.exp(output)
            score_all.extend(output.cpu().detach().numpy())
            test_labels.extend(data.y.cpu().detach().numpy())
        score_all = np.vstack(score_all)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(test_labels, score_all, pos_label=1)
        test_auc = metrics.auc(fpr, tpr)
        report = classification_report(test_labels, y_pre, output_dict=True)
        df = pd.DataFrame(report).transpose()
        pre = df.at['1', 'precision']
        f1 = df.at['1', 'f1-score']
        rec = df.at['1', 'recall']
        acc = accuracy_score(test_labels, y_pre)
        return acc, test_auc, pre, rec, f1

    model = RNA2vec_KNFP_GCN_2layer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    train_loader = DataLoader(train_graphs, batch_size=batch_size)
    f = open("all_" + "auc.txt", "a")
    f11 = open("all_" + "acc.txt", "a")
    f2 = open("all_" + "pre.txt", "a")
    f3 = open("all_" + "recall.txt", "a")
    f4 = open("all_" + "f1.txt", "a")
    data_len = len(train_graphs)
    best_auc = test_acc = test_auc = pre = rec = f1 = 0
    for epoch in range(0, 50):
        loss = train(train_loader, data_len)
        train_acc, train_auc, _, _, _ = test(train_loader)
        test_acc, test_auc, pre, rec, f1 = test(test_loader)
        print(' Epoch:{:03d}, Loss: {:.5f}, Train Acc: {:.5f}, Test Acc: {:.5f} ,Test AUC: {:.5f}'.
              format(epoch, loss, train_acc, test_acc, test_auc))
        print(str(pre) + " " + str(rec) + " " + str(f1))

    f.write(protein + ":" + str(test_auc) + '\n')
    f11.write(protein + ":" + str(test_acc) + '\n')
    f2.write(protein + ":" + str(pre) + '\n')
    f3.write(protein + ":" + str(rec) + '\n')
    f4.write(protein + ":" + str(f1) + '\n')
