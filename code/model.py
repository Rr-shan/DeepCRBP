from torch import nn
import torch
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, TopKPooling, SAGEConv, GATConv, GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp


class RNA2vec_KNFP_GCN_2layer(torch.nn.Module):
    def __init__(self):
        super(RNA2vec_KNFP_GCN_2layer, self).__init__()
        self.conv1 = GraphConv(4, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)

        # sequence
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=3)
        self.conv1d01 = nn.Conv1d(in_channels=84, out_channels=128, kernel_size=3)
        self.lstm = nn.GRU(128, 120, 1, bidirectional=True, batch_first=True)

        # FC
        self.dense_1 = nn.Linear(9616, 102)
        self.dense_3 = nn.Linear(102, 2)

    def forward(self, data):
        target = data.target
        target01 = data.target01

        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 平均池化、最大池化

        x = x1 + x2

        # sequence
        xt = F.relu(self.conv1d(target))
        xl = F.relu(self.conv1d01(
            target01))
        xt = torch.cat((xt, xl), -1)
        xt = F.avg_pool1d(xt, 5)
        xt = F.dropout(xt, p=0.5, training=self.training)
        xt, _ = self.lstm(xt.transpose(1, 2))
        xt = F.dropout(xt, p=0.5, training=self.training)

        # flatten
        xt = xt.reshape((xt.shape[0], -1))
        xc = torch.cat((x, xt), 1)
        xc = F.relu(self.dense_1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = F.log_softmax(self.dense_3(xc), dim=-1)

        return xc


class KNFP_GCN_2layer(torch.nn.Module):
    def __init__(self):
        super(KNFP_GCN_2layer, self).__init__()
        self.conv1 = GraphConv(4, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)

        # sequence
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=3)
        self.conv1d01 = nn.Conv1d(in_channels=84, out_channels=128, kernel_size=3)
        self.lstm = nn.GRU(128, 120, 1, bidirectional=True, batch_first=True)
        # FC
        self.dense_1 = nn.Linear(4816, 102)
        self.dense_3 = nn.Linear(102, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target01 = data.target01

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 平均池化、最大池化

        x = x1 + x2

        xl = F.relu(self.conv1d01(
            target01))

        xt = xl
        xt = F.avg_pool1d(xt, 5)
        xt = F.dropout(xt, p=0.5, training=self.training)
        xt, _ = self.lstm(xt.transpose(1, 2))
        xt = F.dropout(xt, p=0.5, training=self.training)

        # flatten
        xt = xt.reshape((xt.shape[0], -1))
        xc = torch.cat((x, xt), 1)

        xc = F.relu(self.dense_1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = F.log_softmax(self.dense_3(xc), dim=-1)

        return xc


class RNA2vec_GCN_2layer(torch.nn.Module):
    def __init__(self):
        super(RNA2vec_GCN_2layer, self).__init__()
        self.conv1 = GraphConv(4, 128)

        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.conv4 = GraphConv(128, 128)

        # sequence
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=3)
        self.conv1d01 = nn.Conv1d(in_channels=84, out_channels=128, kernel_size=3)
        self.lstm = nn.GRU(128, 120, 1, bidirectional=True, batch_first=True)
        # FC
        self.dense_1 = nn.Linear(4816, 102)
        self.dense_3 = nn.Linear(102, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target = data.target

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)  # 平均池化、最大池化

        x = x1 + x2
        xt = F.relu(self.conv1d(target))
        xt = F.avg_pool1d(xt, 5)
        xt = F.dropout(xt, p=0.5, training=self.training)
        xt, _ = self.lstm(xt.transpose(1, 2))
        xt = F.dropout(xt, p=0.5, training=self.training)

        # flatten
        xt = xt.reshape((xt.shape[0], -1))
        xc = torch.cat((x, xt), 1)

        xc = F.relu(self.dense_1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = F.log_softmax(self.dense_3(xc), dim=-1)

        return xc


class RNA2vec_KNFP(torch.nn.Module):
    def __init__(self):
        super(RNA2vec_KNFP, self).__init__()
        # sequence
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=3)
        self.conv1d01 = nn.Conv1d(in_channels=84, out_channels=128, kernel_size=3)
        self.lstm = nn.GRU(128, 120, 1, bidirectional=True, batch_first=True)  #
        # FC
        self.dense_1 = nn.Linear(9360, 102)
        self.dense_3 = nn.Linear(102, 2)

    def forward(self, data):
        target = data.target
        target01 = data.target01

        xt = F.relu(self.conv1d(target))
        xl = F.relu(self.conv1d01(target01))
        xt = torch.cat((xt, xl), -1)

        xt = F.avg_pool1d(xt, 5)
        xt = F.dropout(xt, p=0.5, training=self.training)
        xt, _ = self.lstm(xt.transpose(1, 2))
        xt = F.dropout(xt, p=0.5, training=self.training)

        # flatten
        xt = xt.reshape((xt.shape[0], -1))
        xc = xt

        xc = F.relu(self.dense_1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = F.log_softmax(self.dense_3(xc), dim=-1)

        return xc


class KNFP_GCN(torch.nn.Module):
    def __init__(self):
        super(KNFP_GCN, self).__init__()
        self.conv1 = GraphConv(4, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.conv3 = GraphConv(128, 128)
        self.pool3 = TopKPooling(128, ratio=0.8)
        self.conv4 = GraphConv(128, 128)

        # sequence
        self.conv1d = nn.Conv1d(in_channels=30, out_channels=128, kernel_size=3)
        self.conv1d01 = nn.Conv1d(in_channels=84, out_channels=128, kernel_size=3)
        self.lstm = nn.GRU(128, 120, 1, bidirectional=True, batch_first=True)
        # FC
        self.dense_1 = nn.Linear(4816, 102)
        self.dense_3 = nn.Linear(102, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        target01 = data.target01

        x1 = F.relu(self.conv1(x, edge_index))
        x2 = F.relu(self.conv2(x1, edge_index))
        x3 = F.relu(self.conv3(x2, edge_index))
        x3 = x3 + x1
        x4 = F.relu(self.conv3(x3, edge_index))
        x = x4
        x = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        xl = F.relu(self.conv1d01(
            target01))
        xt = xl

        xt = F.avg_pool1d(xt, 5)
        xt = F.dropout(xt, p=0.5, training=self.training)
        xt, _ = self.lstm(xt.transpose(1, 2))
        xt = F.dropout(xt, p=0.5, training=self.training)

        # flatten
        xt = xt.reshape((xt.shape[0], -1))
        xc = torch.cat((x, xt), 1)
        xc = F.relu(self.dense_1(xc))
        xc = F.dropout(xc, p=0.5, training=self.training)
        xc = F.log_softmax(self.dense_3(xc), dim=-1)

        return xc


class GCN_2layer(torch.nn.Module):
    def __init__(self):
        super(GCN_2layer, self).__init__()

        self.conv1 = GraphConv(4, 128)
        self.pool1 = TopKPooling(128, ratio=0.8)
        self.conv2 = GraphConv(128, 128)
        self.pool2 = TopKPooling(128, ratio=0.8)
        self.lin1 = torch.nn.Linear(256, 128)
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        x = x1 + x2
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x
