# _*_ coding: utf-8 _*_
"""
Time:     2022-04-09 11:42
Author:   Haolin Yan(XiDian University)
File:     split.py
切分数据集为5折，重排序生成5个json文件
"""

import numpy as np
from sklearn.model_selection import train_test_split
import json
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, default='../data/CVPR_2022_NAS_Track2_train.json')
parser.add_argument("-o", type=str, default="../data")
args = parser.parse_args()

name_list = ['cplfw_rank',
             'market1501_rank',
             'dukemtmc_rank',
             'msmt17_rank',
             'veri_rank',
             'vehicleid_rank',
             'veriwild_rank',
             'sop_rank']
archi_name = "arch"

if __name__ == "__main__":
    path = args.path
    with open(path, 'r') as f:
        data = json.load(f)

    indices = np.arange(len(data.keys()))
    for i in range(5):
        np.random.seed(i)
        train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=i)
        train_data = dict([(archi_name + str(j + 1), data[archi_name + str(j + 1)]) for j in train_indices])
        val_data = dict([(archi_name + str(j + 1), data[archi_name + str(j + 1)]) for j in val_indices])
        with open(args.o + "/data-cv5-train%d.json" % (i + 1), "w") as f:
            json.dump(train_data, f)
        with open(args.o + "/data-cv5-val%d.json" % (i + 1), "w") as f:
            json.dump(val_data, f)
