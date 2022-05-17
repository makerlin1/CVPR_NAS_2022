# _*_ coding: utf-8 _*_
"""
Time:     2022-05-12 8:50
Author:   Haolin Yan(XiDian University)
File:     ensemble.py
"""
from torchensemble.utils import logging
import nni
import torch
import numpy as np
import random
from dnn import Kendall_tau_loss
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
logger = logging.set_logger("log")
from torchensemble import (
                           VotingRegressor,
                           SnapshotEnsembleRegressor,
                           BaggingRegressor,
                           FusionRegressor)

Ensemble = {
            "vote": VotingRegressor,
            "snap": SnapshotEnsembleRegressor,
            "bag": BaggingRegressor,
            "fusion": FusionRegressor}

from DCN import Emlp, NNBaseline
# import torch.nn as nn
import sys
import numpy as np
import torch
sys.path.append("/tmp/pycharm_project_513")
from utils import read_labeled_data
from dataset import MyDataset
import scipy.stats
import argparse
parser = argparse.ArgumentParser("ensemble")
parser.add_argument("-cv", type=int)
parser.add_argument("-cls", type=int)
args = parser.parse_args()


# set the hyparam
# params = {"batch_size": 8,
#           "dim": 16,
#           "model": "embcnn",
#           "lr": 0.1,
#           "eps": 12,
#           "method": "snap",
#           "num_estimators": 3
#           }
params = nni.get_next_parameter()

MODELLIST = {"emlp": Emlp,
             "embcnn": NNBaseline}

cv = args.cv
cls = args.cls

# set dataset
data_train = read_labeled_data('../data/data-cv5-train%d.json' % cv, cls)
data_val = read_labeled_data('../data/data-cv5-val%d.json' % cv, cls)
X_train, y_train = np.array([x[0] for x in data_train]), np.array([x[1] for x in data_train])
X_val, y_val = np.array([x[0] for x in data_val]), np.array([x[1] for x in data_val])
tr_dataset = MyDataset(X_train, y_train)
te_dataset = MyDataset(X_val, y_val)
train_loader = torch.utils.data.DataLoader(tr_dataset,
                                           batch_size=params["batch_size"],
                                           shuffle=True,
                                           num_workers=3,
                                           pin_memory=True)

test_loader = torch.utils.data.DataLoader(te_dataset,
                                           batch_size=params["batch_size"],
                                           shuffle=False,
                                           num_workers=3,
                                           pin_memory=True)

# set model
model = Ensemble[params["method"]](
    estimator=MODELLIST[params["model"]],
    n_estimators=params["num_estimators"],
    estimator_args=params,
    cuda=True,
)

# set optimer
model.set_optimizer('Adam', lr=params["lr"], weight_decay=5e-4)
model.set_criterion(Kendall_tau_loss)

model.fit(train_loader,
          epochs=params["eps"],
          test_loader=test_loader,
          save_model=True,
          save_dir=None)

pred = model.predict(te_dataset.X).flatten()
tekdl = scipy.stats.kendalltau(pred, te_dataset.Y).correlation

pred = model.predict(tr_dataset.X).flatten()
trkdl = scipy.stats.kendalltau(pred, tr_dataset.Y).correlation

nni.report_intermediate_result(trkdl)
nni.report_final_result(tekdl)


