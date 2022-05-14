# _*_ coding: utf-8 _*_
"""
Time:     2022-05-12 16:26
Author:   Haolin Yan(XiDian University)
File:     sort.py
"""
import torch
from dnn import CMPBaseline
import json
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


model = CMPBaseline()
model.load_state_dict(torch.load())

cls = 0
cv = 1
dataset_path = '../data/data-cv5-val%d.json' % cv
with open(dataset_path) as f:
    ds = json.load(f)

X = {}
label = []
for key in ds.keys():
    arch = ds[key]['arch']
    for idx, name in enumerate(RANK_NAME):
        if idx != cls:
            continue
        X[arch] = np.array(convert_X(arch))
        label.append(ds[key][name])


def sort(v):
    if len(v) <= 1:
        return v
    c = v[0]
    left, right = [], []
    for e in v[1:]:
        rx = np.array(X[c] - X[e]).astype(np.float32)
        rx = torch.from_numpy(rx).unsqueeze(0)
        if torch.argmax(model(rx)) == 1:
            left.append(e)
        else:
            right.append(e)
    return sort(left) + [c] + sort(right)


input = list(X.keys())
sorted_v_dict = dict((v, i) for i, v in enumerate(sort(input)))
result = [sorted_v_dict[v] for v in input]
print(scipy.stats.kendalltau(result, label).correlation)
