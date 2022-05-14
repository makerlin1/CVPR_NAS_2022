# _*_ coding: utf-8 _*_
"""
Time:     2022-05-11 23:12
Author:   Haolin Yan(XiDian University)
File:     dnn.py
"""
import torch
import torch.nn as nn


class CMPBaseline(nn.Module):
    def __init__(self, embedding_dim=8):
        super(CMPBaseline, self).__init__()
        self.embedding = nn.Embedding(10, embedding_dim)  # (bs, 25, 64)
        self.loc_embedding = nn.Embedding(25, embedding_dim)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.out_layer = nn.Sequential(nn.Linear(embedding_dim, embedding_dim),
                                       nn.BatchNorm1d(embedding_dim))

        self.head_proj = nn.Sequential(nn.Linear(embedding_dim, 1),
                                       nn.Sigmoid())

    def forward(self, x):
        x1, x2 = x[:, 0, :], x[:, 1, :]
        loc = torch.arange(25).reshape([1, 25]).to(x1.device)
        x1 = self.emb_norm(self.embedding(x1) + self.loc_embedding(loc))
        x1 = x1.permute([0, 2, 1]).contiguous()
        h1 = self.avg_pool(x1).squeeze(-1)
        h1 = self.out_layer(h1)

        x2 = self.emb_norm(self.embedding(x2) + self.loc_embedding(loc))
        x2 = x2.permute([0, 2, 1]).contiguous()
        h2 = self.avg_pool(x2).squeeze(-1)
        h2 = self.out_layer(h2)

        h = (h1 - h2) / 2.
        return self.head_proj(h)


class NNBaseline(nn.Module):
    def __init__(self,
                 embedding_dim=32):
        super(NNBaseline, self).__init__()
        # 编码器
        self.embedding = nn.Embedding(10, embedding_dim)  # (bs, 25, 64)
        self.loc_embedding = nn.Embedding(25, embedding_dim)
        self.emb_norm = nn.LayerNorm(embedding_dim)
        self.convbnlayer = nn.Sequential(nn.Conv1d(embedding_dim,
                                                   1,
                                                   kernel_size=7,
                                                   stride=1,
                                                   padding=7 // 2),
                                         nn.LayerNorm([1, 25]),
                                         nn.Softsign())  # (bsize, 1, 25)

        self.out_layer = nn.Sequential(nn.Linear(embedding_dim * 2, 1),
                                       nn.BatchNorm1d(1))
        self.neck = nn.Sequential(nn.Conv1d(embedding_dim, embedding_dim * 2, kernel_size=1),
                                  nn.BatchNorm1d(embedding_dim * 2),
                                  nn.Softsign())

        self.avg_pool = nn.AdaptiveAvgPool1d(1)  # (bsize, 32, 25) -> (bsize, 32, 1)

    def forward(self, x):
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        x = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))
        x = x.permute([0, 2, 1]).contiguous()
        weight = self.convbnlayer(x)
        x = x * weight
        x = self.neck(x)
        h = self.avg_pool(x).squeeze(-1)
        return self.out_layer(h)


class FGM():
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=-0.3, emb_name="embedding"):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embedding'):
        # emb_name这个参数要换成你模型中embedding的参数名
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class MixLoss(nn.Module):
    def __init__(self):
        super(MixLoss, self).__init__()
        self.kdl_loss = Kendall_tau_loss()
        self.mse = nn.MSELoss()

    def forward(self, pred, y):
        mse_l = self.mse(pred, y)
        kdl_l = self.kdl_loss(pred, y)
        return mse_l + kdl_l


class Kendall_tau_loss(nn.Module):
    def __init__(self):
        super(Kendall_tau_loss, self).__init__()
        self.soft_sign = nn.Softsign()

    def forward(self, pred, y):
        """
        pred: (bs,)
        y: (bs,)
        """
        pred = pred.flatten()
        y = y.flatten()
        # raise ValueError("{},{}".format(pred.shape, y.shape))
        N = len(pred)
        num = N * (N - 1) / 2
        mask = torch.triu(torch.ones([N, N])).to(pred.device)
        rx = pred.unsqueeze(0) - pred.unsqueeze(1)  # (bs, bs)
        rx = self.soft_sign(mask * rx).flatten()
        ry = y.unsqueeze(0) - y.unsqueeze(1)  # (bs, bs)
        ry = torch.sgn(mask * ry).flatten()
        kdl = torch.sum(rx * ry) / num
        return -kdl


if __name__ == "__main__":
    model = CMPBaseline()
    x1 = torch.ones(2, 25)
    x2 = torch.ones(2, 25)*2

    print(model(x2.long(), x1.long()))