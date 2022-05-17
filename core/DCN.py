# _*_ coding: utf-8 _*_
"""
Time:     2022-05-12 20:24
Author:   Haolin Yan(XiDian University)
File:     DCN.py
"""
import torch
import torch.nn as nn
from collections import OrderedDict


class NNBaseline(nn.Module):
    def __init__(self, **kargs):
        super(NNBaseline, self).__init__()
        embedding_dim = kargs["dim"]
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
        x = x.long()
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        x = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))
        x = x.permute([0, 2, 1]).contiguous()
        weight = self.convbnlayer(x)
        x = x * weight
        x = self.neck(x)
        h = self.avg_pool(x).squeeze(-1)
        return self.out_layer(h)


class CRmlp(nn.Module):
    def __init__(self, **kwargs):
        super(CRmlp, self).__init__()
        dim = kwargs["dim"]
        depth = kwargs["depth"]
        self.embedding = nn.Embedding(10, dim)
        self.loc_embedding = nn.Embedding(25, dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.neck = nn.ModuleList()
        for _ in range(depth):
            self.neck.append(Correlation_Block(num_features=25, dim=dim))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(dim, 1),
                                nn.BatchNorm1d(1))

    def forward(self, x):
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        h = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))  # (N, 25, dim)
        for ops in self.neck:
            h = ops(h)
        h = self.avgpool(h.permute([0, 2, 1]).contiguous()).squeeze(-1)
        return self.fc(h)


class TRmlp(nn.Module):
    def __init__(self, **kwargs):
        super(TRmlp, self).__init__()
        dim = kwargs["dim"]
        depth = kwargs["depth"]
        self.embedding = nn.Embedding(10, dim)
        self.loc_embedding = nn.Embedding(25, dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.neck = nn.ModuleList()
        for _ in range(depth):
            self.neck.append(nn.TransformerEncoderLayer(d_model=dim, nhead=8))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(dim, 1),
                                nn.BatchNorm1d(1))

    def forward(self, x):
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        h = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))  # (N, 25, dim)
        for ops in self.neck:
            h = ops(h)
        h = self.avgpool(h.permute([0, 2, 1]).contiguous()).squeeze(-1)
        return self.fc(h)


class Emlp(nn.Module):
    def __init__(self, **kwargs):
        super(Emlp, self).__init__()
        dim = kwargs["dim"]
        self.embedding = nn.Embedding(10, dim)
        self.loc_embedding = nn.Embedding(25, dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(dim, 1),
                                nn.BatchNorm1d(1))

    def forward(self, x):
        x = x.long()
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        h = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))  # (N, 25, dim)
        h = self.avgpool(h.permute([0, 2, 1]).contiguous()).squeeze(-1)
        return self.fc(h)


class xDeepFM(nn.Module):
    def __init__(self, **kwargs):
        super(xDeepFM, self).__init__()
        dim = kwargs["dim"]
        depth = kwargs["depth"]
        self.embedding = nn.Embedding(10, dim)
        self.loc_embedding = nn.Embedding(25, dim)
        self.emb_norm = nn.LayerNorm(dim)
        self.cin = CIN(input_dim=25, num_layers=depth)
        self.fc = nn.Sequential(nn.Linear(25, dim),
                                nn.BatchNorm1d(dim),
                                nn.Softsign(),
                                nn.Linear(dim, 1),
                                nn.BatchNorm1d(1))

    def forward(self, x):
        loc = torch.arange(25).reshape([1, 25]).to(x.device)
        h = self.emb_norm(self.embedding(x) + self.loc_embedding(loc))  # (N, 25, dim)
        return self.cin(h) + self.fc(x.float())


class SimpleMLP(nn.Module):
    def __init__(self, **args):
        super(SimpleMLP, self).__init__()
        dim = args["dim"]
        self.fc = nn.Sequential(nn.Linear(25, dim),
                                nn.BatchNorm1d(dim),
                                nn.Softsign(),
                                nn.Linear(dim, 1),
                                nn.BatchNorm1d(1))

    def forward(self, x):
        x = x.float()
        return self.fc(x)


class CIN(torch.nn.Module):
    def __init__(self, input_dim, num_layers=2):
        super(CIN, self).__init__()
        # CIN 网络有几层，也就是要几阶
        self.num_layers = num_layers
        # 一维卷积层
        self.conv_layers = torch.nn.ModuleList()
        fc_input_dim = 0
        for i in range(self.num_layers):
            ''' in_channels: 输入信号的通道 向量的维度 ,input_dim的长度指的是特征的总数
                out_channels:卷积产生的通道。有多少个out_channels，就需要多少个1维卷积 
                kerner_size :卷积核的尺寸，卷积核的大小为(k,)，第二个维度是由in_channels来决定的，所以实际上卷积大小为kerner_size*in_channels
                stride : 卷积步长 
                dilation :卷积核元素之间的间距'''
            self.conv_layers.append(

                torch.nn.Conv1d(in_channels=input_dim * input_dim, out_channels=input_dim, kernel_size=1,
                                stride=1, dilation=1, bias=True))
            fc_input_dim += input_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)
        self.softsgn = nn.Softsign()

    def forward(self, x):
        xs = list()
        '''举例  x.shape = [1,22,16] 1表示batch_size,表示有几维数据，22表示特征的维数，16是embedding层的向量大小
        经过 x.unsqueeze(2)后 x.shape = [1,22,1,16]
        经过 x.unsqueeze(1)后 x.shape = [1,1,22,16]  
        x.unsqueeze(2) * x.unsqueeze(1) 后   x.shape =[1,22,22,16]
        进过卷积层后变为 x.shape =[1,16,16]
        经过 sum pooling  变为 1维
         '''
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            h1 = h.unsqueeze(1)
            x = x0 * h1
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = self.softsgn(self.conv_layers[i](x))
            h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


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

        block_list = [3, 5, 3]
        dim_list = [embedding_dim]
        ratio_list = [0.5, 0.5, 0.5]
        num_features = [25, 25, 25]
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

        self.conv2 = nn.Sequential(nn.Conv1d(25, 1, kernel_size=1),
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
        z = z0 + z1 + z2
        v = self.conv2(z).squeeze(1)
        return self.output_fc(v)


if __name__ == "__main__":
    model = SimpleMLP(32)
    x = torch.ones(8, 25).long()
    print(model(x).shape)
