# _*_ coding: utf-8 _*_
"""
Time:     2022-04-09 10:06
Author:   Haolin Yan(XiDian University)
File:     evaluate.py
模型评估脚本：
输入对应的预测结果的前缀名(e.g. gpnas-1.json 对应 gpnas-),
输入对应的真实结果的前缀名(e.g. data-cv5-val1.json 对应 data-cv5-val),
注意5个文件的序号必须是从1~5
例如：有以下label
data-cv5-val1.json
data-cv5-val2.json
data-cv5-val3.json
data-cv5-val4.json
data-cv5-val5.json
有以下预测结果
gpnas-1.json
gpnas-2.json
gpnas-3.json
gpnas-4.json
gpnas-5.json
命令：
python ../utils/evaluate.py -pred gpnas-  -label data-cv5-val
"""
import scipy.stats
import argparse
import json
from tabulate import tabulate
import numpy as np


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
    parser = argparse.ArgumentParser()
    parser.add_argument("-pred", type=str, help="val_pred.json", default="../data/data-cv5-val")
    parser.add_argument("-label", type=str, help="val_label.json", default="../data/data-cv5-val")
    args = parser.parse_args()
    CV_record = []
    headers = name_list + ["avg score"]
    for i in range(1, 6):
        with open(args.pred + str(i) + ".json", 'r') as f:
            pred = json.load(f)

        with open(args.label + str(i) + ".json", 'r') as f:
            target = json.load(f)

        pred_list = [[], [], [], [], [], [], [], []]
        target_list = [[], [], [], [], [], [], [], []]

        for v in pred.values():
            for j, k in enumerate(name_list):
                pred_list[j].append(v[k])

        for v in target.values():
            for j, k in enumerate(name_list):
                target_list[j].append(v[k])

        record = []
        for j, k in enumerate(name_list):
            kdl = scipy.stats.kendalltau(pred_list[j], target_list[j])
            record.append(kdl.correlation)
        record.append(np.mean(record))
        CV_record.append(record)
        print(tabulate([record], headers=headers, tablefmt="grid"))

    CV_record = np.mean(np.array(CV_record), axis=0)
    print(tabulate([CV_record], headers=headers, tablefmt="grid"))







