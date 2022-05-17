# _*_ coding: utf-8 _*_
"""
Time:     2022-05-15 20:29
Author:   Haolin Yan(XiDian University)
File:     predict.py
"""
import json
from dataset import convert_X
from DCN import Emlp
import torch
import sys
import numpy as np
import scipy.stats
import pandas as pd
param_path = "/tmp/pycharm_project_513/core/params/cls7emlp_best.pth"
path = '../data/submit.json'
# path = "submit_2.json"
output_path = "submit_2.json"
# class_ = "msmt17_rank"
# class_ = "veri_rank"
# class_ = "vehicleid_rank"
# class_ = "veriwild_rank"
class_ = "sop_rank"

model = Emlp(dim=128)
model.load_state_dict(torch.load(param_path))
model.eval()
# pred = []
# pred = pd.Series(pred).rank().values
# print(pred)
# print(scipy.stats.kendalltau(pred, y_val))
with open(path) as f:
    ds = json.load(f)

pred_X = []
key_X = []
for key in ds.keys():
    key_X.append(key)
    pred_X.append(convert_X(ds[key]['arch']))

input_list = [pred_X[i:i + 500] for i in range(0, len(pred_X), 500)]
rank_all = []
pred = []
for x_data in input_list:
    x = torch.LongTensor(x_data)
    y = model(x).flatten().detach().numpy()
    pred += y.tolist()
# pred = pd.Series(pred).rank().values
for k, pred_y in zip(key_X, pred):
    ds[k][class_] = int(pred_y)

with open(output_path, 'w') as f:
    json.dump(ds, f)

