# _*_ coding: utf-8 _*_
"""
Time:     2022-05-12 8:50
Author:   Haolin Yan(XiDian University)
File:     ensemble.py
"""
from torchensemble.utils import logging
logger = logging.set_logger("log")
# print(torchensemble.__file__)
from torchensemble import GradientBoostingRegressor
from dnn import CMPBaseline
import torch.nn as nn
import sys
import numpy as np
import torch
sys.path.append("/tmp/pycharm_project_513")
from utils import read_labeled_data
from dataset import CMPDataset
import scipy.stats

cv = 2
cls = 0
data_train = read_labeled_data('../data/data-cv5-train%d.json' % cv, cls)
data_val = read_labeled_data('../data/data-cv5-val%d.json' % cv, cls)
X_train, y_train = np.array([x[0] for x in data_train]), np.array([x[1] for x in data_train])
X_val, y_val = np.array([x[0] for x in data_val]), np.array([x[1] for x in data_val])
tr_dataset = CMPDataset(X_train, y_train)
te_dataset = CMPDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(tr_dataset,
                                           batch_size=8,
                                           shuffle=True,
                                           num_workers=3,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(te_dataset,
                                           batch_size=8,
                                           shuffle=False,
                                           num_workers=3,
                                           pin_memory=True)
model = GradientBoostingRegressor(
    estimator=CMPBaseline,
    n_estimators=8,
    cuda=True,
)

model.set_optimizer('Adam', lr=0.01, weight_decay=5e-4)

# Training
model.fit(train_loader=train_loader, test_loader=test_loader, epochs=64)                 # the number of training epochs

