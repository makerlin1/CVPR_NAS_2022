# _*_ coding: utf-8 _*_
"""
Time:     2022-05-11 16:39
Author:   Haolin Yan(XiDian University)
File:     loaddata.py
"""
import json
import numpy as np

DEPTH = {"j": 10., "k": 11., "l": 12.}
NUM_HEADS = {"1": 12., "2": 11., "3": 10.}
MLP_RATIO = {"1": 4.0, "2": 3.5, "3": 3.0}
name_list = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank', 'msmt17_rank', 'veri_rank', 'vehicleid_rank',
             'veriwild_rank', 'sop_rank']


def pad(x, l=10):
    while len(x) < l:
        x += [0.]
    return x


def convert(X):
    v = [DEPTH[X[0]]]
    X = X[1:]
    num_head = []
    mlp_ratio = []
    for i in range(36):
        id = (i + 1) % 3
        if id == 0:
            continue
        elif id == 1:
            _ = NUM_HEADS.get(X[i], False)
            if _:
                num_head.append(_)
        else:
            _ = MLP_RATIO.get(X[i], False)
            if _:
                mlp_ratio.append(_)
    v += pad(num_head, 12) + pad(mlp_ratio, 12)
    return v


def load_training_data(train_json, cls):
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    train_list = []
    arch_list_train = []
    for key in train_data.keys():
        for idx, name in enumerate(name_list):
            if idx != cls:
                continue
            train_list.append(train_data[key][name])
        arch_list_train.append(convert(train_data[key]['arch']))
    return np.array(arch_list_train), np.array(train_list)


def load_test_data(test_json):
    with open(test_json, 'r') as f:
        test_data = json.load(f)
    arch_list_test = []
    for key in test_data.keys():
        arch_list_test.append(convert(test_data[key]['arch']))
    return np.array(arch_list_test)


def read_labeled_data(train_json, cls):
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    data = []
    for key in train_data.keys():
        for idx, name in enumerate(name_list):
            if idx != cls:
                continue
            y = train_data[key][name]
        data.append([train_data[key]['arch'], y])
    return data


def read_unlabeled_data(train_json, cls):
    with open(train_json, 'r') as f:
        train_data = json.load(f)
    data = []
    for key in train_data.keys():
        data.append([train_data[key]['arch'], None, 0, 0, None, None])
    return data


class AverageMeter:
    """
    Computes and stores the average and current value.

    Parameters
    ----------
    name : str
        Name to display.
    fmt : str
        Format string to print the values.
    """

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        """
        Reset the meter.
        """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """
        Update with value and weight.

        Parameters
        ----------
        val : float or int
            The new value to be accounted in.
        n : int
            The weight of the new value.
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = '{name}: {avg' + self.fmt + '}'
        return fmtstr.format(**self.__dict__)
