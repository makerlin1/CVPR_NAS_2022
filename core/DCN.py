# _*_ coding: utf-8 _*_
"""
Time:     2022-05-12 20:24
Author:   Haolin Yan(XiDian University)
File:     DCN.py
"""
import torch
import torch.nn as nn
from collections import OrderedDict


# import torchsort


# def spearmanr(pred, target, **kw):
#     pred = torchsort.soft_rank(pred, **kw)
#     target = torchsort.soft_rank(target, **kw)
#     pred = pred - pred.mean()
#     pred = pred / pred.norm()
#     target = target - target.mean()
#     target = target / target.norm()
#     return (pred * target).sum()


class SimpleMLP(nn.Module):
    def __init__(self, hidden_unit=50000):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(nn.Linear(25, hidden_unit),
                                nn.BatchNorm1d(hidden_unit),
                                nn.Softsign(),
                                nn.Linear(hidden_unit, 1))

    def forward(self, x):
        # x = x.float()
        return self.fc(x)


# class XDeepFM(nn.Module):
#     def __init__(self, dim):
#         super(XDeepFM, self).__init__()
#         self.fc = SimpleMLP(hidden_unit=128)


# class SpearmanLoss(nn.Module):
#     def __init__(self):
#         super(SpearmanLoss, self).__init__()
#         self.mse = nn.MSELoss()
#
#     def forward(self, pred, label):
#         l_mse = self.mse(pred, label)
#         l_s = spearmanr(pred.view(1, -1), label.view(1, -1))
#         return l_s + l_mse


class Correlation_Block(nn.Module):
    def __init__(self, num_features, dim, ratio=2, act=nn.Softsign):
        super(Correlation_Block, self).__init__()
        self.proj_mlp0 = nn.Sequential(nn.Linear(dim, int(dim * ratio)),
                                       nn.BatchNorm1d(num_features))
        self.proj_mlp1 = nn.Sequential(nn.Linear(int(dim * ratio), dim),
                                       nn.BatchNorm1d(num_features))
        self.act = act()
        self.feed_norm = nn.BatchNorm1d(num_features)
        self.proj_conv = nn.Conv1d(num_features, num_features, kernel_size=1)
        self.output_norm = nn.BatchNorm1d(num_features)

    def forward(self, x):
        """x:(N, num_features, dim)"""
        v = self.proj_mlp0(x)  # (N, num_f, dim*r)
        w = self.act(torch.bmm(v, v.permute([0, 2, 1])))  # (N, num_f, num_f)
        v = torch.bmm(w, v)  # (N, num_f, dim*r)
        v = self.proj_mlp1(v)  # (N, num_f, dim)
        x = self.feed_norm(v + x)
        v = self.proj_conv(x)
        return self.output_norm(v + x)


class DeepCorNN(nn.Module):
    def __init__(self, embedding_dim):
        super(DeepCorNN, self).__init__()
        self.embedding = nn.Embedding(10, embedding_dim)
        self.loc_embedding = nn.Embedding(25, embedding_dim)
        self.emb_norm = nn.LayerNorm(embedding_dim)

        # block_list = [3, 5, 3]
        block_list = [1, 1, 1]
        dim_list = [embedding_dim]
        ratio_list = [0.5, 0.5, 0.5]
        num_features = [25, 32, 64]
        # 25,dim
        self.Stage0 = nn.Sequential(OrderedDict([("CB_%d" % j,
                                                  Correlation_Block(num_features[0], dim_list[0], ratio_list[0]))
                                                 for j in range(block_list[0])]))
        self.conv0 = nn.Sequential(nn.Conv1d(num_features[0], num_features[1], kernel_size=1),
                                   nn.BatchNorm1d(num_features[1]),
                                   nn.Softsign())

        # 32,dim
        self.Stage1 = nn.Sequential(OrderedDict([("CB_%d" % j,
                                                  Correlation_Block(num_features[1], dim_list[0], ratio_list[1]))
                                                 for j in range(block_list[1])]))
        self.conv1 = nn.Sequential(nn.Conv1d(num_features[1], num_features[2], kernel_size=1),
                                   nn.BatchNorm1d(num_features[2]),
                                   nn.Softsign())

        # 64,dim
        self.Stage2 = nn.Sequential(OrderedDict([("CB_%d" % j,
                                                  Correlation_Block(num_features[2], dim_list[0], ratio_list[2]))
                                                 for j in range(block_list[2])]))

        self.conv2 = nn.Sequential(nn.Conv1d(sum(num_features), 1, kernel_size=1),
                                   nn.BatchNorm1d(1),
                                   nn.Softsign()
                                   )
        self.output_fc = nn.Sequential(nn.Linear(embedding_dim, 1),
                                       nn.BatchNorm1d(1))

    def forward(self, x):
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        x = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))  # (N, 25, dim)
        z0 = self.Stage0(x)
        h0 = self.conv0(z0)
        z1 = self.Stage1(h0)
        h1 = self.conv1(z1)
        z2 = self.Stage2(h1)
        z = torch.concat([z0, z1, z2], dim=1)
        v = self.conv2(z).squeeze(1)
        return self.output_fc(v)


if __name__ == "__main__":
    model = DeepCorNN(8)
    x = torch.ones(8, 25).long()
    print(model(x).shape)
