# _*_ coding: utf-8 _*_
"""
Time:     2022-05-16 21:19
Author:   Haolin Yan(XiDian University)
File:     merge.py
"""
import json
# target_json = "CVPR_2022_NAS_Track2_submit_A_Shoahon.json"
target_json = "CVPR_2022_NAS_Track2_submit_A_Shaohong_haolin_0.json"
input_json = "submit_first.json"
output_json = "CVPR_2022_NAS_Track2_submit_A_Shaohong_haolin.json"
keylist = ["market1501_rank", "msmt17_rank", "veriwild_rank"]

with open(target_json, "r") as f:
    target = json.load(f)
with open(input_json, "r") as f:
    input_ = json.load(f)


for key in keylist:
    for k in target.keys():
        target[k][key] = input_[k][key]

with open(output_json, "w") as f:
    json.dump(target, f)

