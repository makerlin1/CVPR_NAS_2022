import nni
import torch.optim
import numpy as np
import random
import sys
import os
from DCN import Emlp
from train_model import Trainer, kendall_tau
from dataset_utils import NASDatasetV2, RANK_NAME
from dataset import convert_X
sys.path.append("/tmp/pycharm_project_513")
os.environ["SAVE_DIR"] = "nni"
seed = 2022
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
# 用什么工具产生什么效果，数字
params = nni.get_next_parameter()
cls = 6
# 准备数据
te_dataset = NASDatasetV2(use_ranks=[RANK_NAME[cls]], mode='val', transform=convert_X)
tr_dataset = NASDatasetV2(use_ranks=[RANK_NAME[cls]], mode='train', transform=convert_X)
# data_train = read_labeled_data('../data/data-cv5-train%d.json' % cv, cls)
# data_val = read_labeled_data('../data/data-cv5-val%d.json' % cv, cls)
# X_train, y_train = np.array([x[0] for x in data_train]), np.array([x[1] for x in data_train])
# X_val, y_val = np.array([x[0] for x in data_val]), np.array([x[1] for x in data_val])
# 设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modelcfg = {"dim": params["dim"], "depth": 0}
model = Emlp(**modelcfg).to(device)
model = Trainer(
                device,
                lr=params["lr"],
                modelcfg=modelcfg,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                attack=params["attack"],
                metric=kendall_tau,
                use_ema=params["use_ema"],
                dynamic_lr=False,
                model=model,
                tr_dataset=tr_dataset,
                te_dataset=te_dataset
                )
model.fit()