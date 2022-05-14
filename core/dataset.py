# _*_ coding: utf-8 _*_
"""
Time:     2022-05-11 23:14
Author:   Haolin Yan(XiDian University)
File:     dataset.py
"""
from torch.utils.data import Dataset
import numpy as np

RANK_NAME = ['cplfw_rank',
             'market1501_rank',
             'dukemtmc_rank',
             'msmt17_rank',
             'veri_rank',
             'vehicleid_rank',
             'veriwild_rank',
             'sop_rank']

DEPTH = {"l": 1,
         "j": 2,
         "k": 3}

NUM_HEADs = {"0": 0,
             "1": 4,
             "2": 5,
             "3": 6}

MLP_RATIO = {"0": 0,
             "1": 7,
             "2": 8,
             "3": 9}

Type = {0: NUM_HEADs,
        1: MLP_RATIO}


def convert_X(arch):
    X = [DEPTH[arch[0]]]
    arch = arch[1:]
    assert len(arch) == 36
    for i in range(36):
        id = (i + 1) % 3
        if id == 0:
            continue
        elif id == 1:
            X.append(NUM_HEADs[arch[i]])
        else:
            X.append(MLP_RATIO[arch[i]])

    return X


class MyDataset(Dataset):
    def __init__(self, X, Y=None):
        super(MyDataset, self).__init__()
        if len(X.shape) != 1:
            X = X.flatten()
        # self.X = np.array([convert_X(x) for x in X]).astype(np.int64)
        self.X = np.array([convert_X(x) for x in X]).astype(np.float32)
        self.training = False
        if Y is not None:
            self.Y = np.array(Y).astype(np.float32)
            self.training = True

    def __getitem__(self, index):
        if self.training:
            return self.X[index], np.array([self.Y[index]])
        else:
            return self.X[index], np.array([0.0])

    def __len__(self):
        return len(self.X)


class CMPDataset(Dataset):
    def __init__(self, X ,Y=None):
        super(CMPDataset, self).__init__()
        N = len(X)
        self.X = []
        self.Y = []
        for i in range(N):
            for j in range(N):
                if i > j:
                    l = 1.0
                elif i == j:
                    l = 0.5
                else:
                    l = 0
                self.X.append([convert_X(X[i]), convert_X(X[j])])
                self.Y.append(l)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, item):
        return np.array(self.X[item]).astype(np.int64), np.array([self.Y[item]]).astype(np.float32)

if __name__ == "__main__":
    import sys
    import torch
    import numpy as np
    sys.path.append("/tmp/pycharm_project_513")
    from utils import read_labeled_data
    cv, cls = 1, 0
    data_train = read_labeled_data('../data/data-cv5-train%d.json' % cv, cls)
    data_val = read_labeled_data('../data/data-cv5-val%d.json' % cv, cls)
    X_train, y_train = np.array([x[0] for x in data_train]), np.array([x[1] for x in data_train])
    X_val, y_val = np.array([x[0] for x in data_val]), np.array([x[1] for x in data_val])
    dataset = CMPDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=5,
                                              shuffle=False,
                                              num_workers=3,
                                              pin_memory=True)
    for (x, y) in loader:
        print(x.shape, y.shape)
        break



