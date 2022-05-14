# _*_ coding: utf-8 _*_
"""
Time:     2022-05-11 16:35
Author:   Haolin Yan(XiDian University)
File:     xgb.py
"""
import nni
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import scipy.stats


class XGBoost:
    def __init__(self, config):
        config["random_state"] = 20
        self.estimator = xgb.XGBRegressor(**config)

    def fit(self, X, Y):
        self.estimator.fit(X, Y)
        pred_Y = self.predict(X)
        return mean_squared_error(pred_Y, Y, multioutput="raw_values")

    def predict(self, X):
        return self.estimator.predict(X)


if __name__ == "__main__":
    import sys
    sys.path.append("/tmp/pycharm_project_513")
    from utils import load_training_data, load_test_data
    cls = 1
    X_train, y_train = load_training_data("../data/data-cv5-train3.json", cls=cls)
    X_val, y_val = load_training_data("../data/data-cv5-val3.json", cls=cls)
    X_test = load_test_data("../data/CVPR_2022_NAS_Track2_test.json")
    # search
    params = nni.get_next_parameter()
    # debug
    # import json
    # with open("xgb_cls0.json", "r") as f:
    #     params = json.load(f)
    model = XGBoost(params)
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    kdl = scipy.stats.kendalltau(y_train_pred.flatten(), y_train.flatten()).correlation
    nni.report_intermediate_result(kdl)
    y_val_pred = model.predict(X_val)
    kdl = scipy.stats.kendalltau(y_val_pred.flatten(), y_val.flatten()).correlation
    loss = mean_squared_error(y_val_pred.flatten(), y_val.flatten())
    nni.report_final_result(kdl)
