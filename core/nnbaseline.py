# _*_ coding: utf-8 _*_
"""
Time:     2022-05-11 23:54
Author:   Haolin Yan(XiDian University)
File:     nnbaseline.py
"""
import torch
import math
import torch.nn as nn
from nni.nas.pytorch.utils import AverageMeter
import logging
from torch.utils.tensorboard import SummaryWriter
from dnn import FGM, EMA, Kendall_tau_loss
from DCN import DeepCorNN, SimpleMLP
import os
import nni
import scipy.stats
import inspect


def kendall_tau(output, target, num_classes=1):
    kdl = dict()
    for i in range(num_classes):
        kdl["task-%d" % (i + 1)] = scipy.stats.kendalltau(output[:, i].cpu().detach().numpy(),
                                                          target[:, i].cpu().detach().numpy()).correlation
    return kdl["task-1"]


class Trainer:
    def __init__(self,
                 X_val,
                 y_val,
                 device,
                 lr=0.1,
                 dim=32,
                 batch_size=32,
                 epochs=100,
                 attack=False,
                 metric=kendall_tau,
                 use_ema=False,
                 dynamic_lr=False
                 ):
        args, _, _, values = inspect.getargvalues(inspect.currentframe())
        values.pop("self")

        for arg, val in values.items():
            setattr(self, arg, val)

    def set_lr_schedulr(self, opt):
        warm_up_iter = 10
        lr_min = 3e-7
        lr_max = self.lr
        T_max = self.epochs
        lambda0 = lambda cur_iter: cur_iter / warm_up_iter if cur_iter < warm_up_iter else \
            (lr_min + 0.5 * (lr_max - lr_min) * (
                        1.0 + math.cos((cur_iter - warm_up_iter) / (T_max - warm_up_iter) * math.pi))) / 0.1
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lambda0)

    def build(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.writer = SummaryWriter(os.path.join(os.environ["SAVE_DIR"], 'tensorboard'))

        # self.model = DeepCorNN(self.dim).to(self.device)
        self.model = SimpleMLP(hidden_unit=self.dim).to(self.device)
        self.opt = torch.optim.AdamW(self.model.parameters(),
                                     lr=self.lr)
        self.criterion = nn.MSELoss()
        if self.attack:
            self.fgm = FGM(self.model)
        if self.use_ema:
            # 初始化
            self.ema = EMA(self.model, 0.99)
            self.ema.register()
        self.te_dataset = MyDataset(self.X_val, self.y_val)
        if self.dynamic_lr:
            self.set_lr_schedulr(self.opt)

    def fit(self, X_train, y_train):
        self.build()
        tr_dataset = MyDataset(X_train, y_train)
        best_kdl = -100
        for epoch in range(self.epochs):
            self.model.train()
            train_loss = self.train_one_epoch(epoch, tr_dataset)
            for tag, value in self.model.named_parameters():
                tag = tag.replace('.', '/')
                self.writer.add_histogram(tag, value.data.cpu().numpy(), epoch + 1)
                self.writer.add_histogram(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
            if self.use_ema:
                self.ema.apply_shadow()
            self.model.eval()
            avg_loss, avg_metric = self.validate(epoch, self.te_dataset)
            is_best = best_kdl < avg_metric
            best_kdl = max(best_kdl, avg_metric)
            self.writer.add_scalar("kdl/best", best_kdl, global_step=epoch)
            nni.report_intermediate_result(avg_metric)
            if is_best:
                torch.save(self.model.state_dict(), "best.pth")
            if self.use_ema:
                self.ema.restore()
            if self.dynamic_lr:
                self.scheduler.step()

        nni.report_final_result(best_kdl)
        self.writer.add_hparams({"lr": self.lr,
                                 "dim": self.dim,
                                 "batch_size": self.batch_size,
                                 "epochs": self.epochs,
                                 "attack": self.attack,
                                 "ema": self.use_ema,
                                 "dynamic_lr": self.dynamic_lr
                                 },
                                {"kdl/best": best_kdl})

    def predict(self, X):
        self.model.load_state_dict(torch.load("best.pth"))
        self.model.eval()
        test_dataset = MyDataset(X)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=100,
                                                   shuffle=False,
                                                   num_workers=3,
                                                   pin_memory=True)
        result = []
        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            bs = x.size(0)
            with torch.no_grad():
                logits = self.model(x)
            result += logits.cpu().detach().numpy().flatten().tolist()
        return np.array(result)

    def train_one_epoch(self, epoch, dataset):
        train_loader = torch.utils.data.DataLoader(dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=True,
                                                   num_workers=3,
                                                   pin_memory=True)
        losses = AverageMeter("losses")
        metric = AverageMeter("metric")
        nbatch = len(train_loader)
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            bs = x.size(0)
            self.opt.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            metric.update(self.metric(logits, y))
            if self.attack:
                self.fgm.attack()
                logits_adv = self.model(x)
                loss_adv = self.criterion(logits_adv, y)
                loss_adv.backward()
                self.fgm.restore()
            self.opt.step()
            if self.use_ema:
                self.ema.update()
            losses.update(loss.item(), bs)

        avg_loss = losses.avg
        avg_metric = metric.avg
        self.logger.info("Train: [{:3d}/{}] Avg loss {:.4f}".format(epoch + 1, self.epochs, avg_loss))
        self.writer.add_scalar("loss/train", avg_loss, global_step=epoch)
        self.writer.add_scalar("kdl/train", avg_metric, global_step=epoch)
        return avg_loss

    def validate(self, epoch, dataset):
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  batch_size=100,
                                                  shuffle=True,
                                                  num_workers=3,
                                                  pin_memory=True)
        losses = AverageMeter("losses")
        metric = AverageMeter("metric")
        nbatch = len(test_loader)

        for step, (x, y) in enumerate(test_loader):
            x, y = x.to(self.device, non_blocking=True), y.to(self.device, non_blocking=True)
            bs = x.size(0)
            logits = self.model(x)
            loss = self.criterion(logits, y)
            losses.update(loss.item(), 1)
            # print(logits, y)
            metric.update(self.metric(logits, y))
        avg_loss = losses.avg
        avg_metric = metric.avg
        self.logger.info(
            "VAL: [{:3d}/{}] Avg loss {:.4f} Avg KDL {:.4f}".format(epoch + 1, self.epochs, avg_loss, avg_metric))
        self.writer.add_scalar("loss/val", avg_loss, global_step=epoch)
        self.writer.add_scalar("kdl/test", avg_metric, global_step=epoch)
        # evaluate
        return avg_loss, avg_metric


if __name__ == "__main__":
    import os
    import torch.optim
    import torch.nn as nn
    from dnn import NNBaseline
    from dataset import MyDataset
    import argparse
    import numpy as np
    import random
    import sys
    sys.path.append("/tmp/pycharm_project_513")
    from utils import read_labeled_data
    seed = 2022
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser("train nn")
    parser.add_argument("-o", type=str)
    parser.add_argument("-cv", type=int, default=1)
    parser.add_argument("-cls", type=int, default=0)
    parser.add_argument("-dim", type=int, default=64)
    parser.add_argument("-bs", type=int, default=8)
    parser.add_argument("-eps", type=int, default=512)
    parser.add_argument("-lr", type=float, default=0.1)
    parser.add_argument("-attack", type=bool, default=False)
    parser.add_argument("-use_ema", type=bool, default=False)
    args = parser.parse_args()
    os.environ["SAVE_DIR"] = args.o

    data_train = read_labeled_data('../data/data-cv5-train%d.json' % args.cv, args.cls)
    data_val = read_labeled_data('../data/data-cv5-val%d.json' % args.cv, args.cls)
    X_train, y_train = np.array([x[0] for x in data_train]), np.array([x[1] for x in data_train])
    X_val, y_val = np.array([x[0] for x in data_val]), np.array([x[1] for x in data_val])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Trainer(X_val,
                    y_val,
                    device,
                    lr=args.lr,
                    dim=args.dim,
                    batch_size=args.bs,
                    epochs=args.eps,
                    attack=args.attack,
                    metric=kendall_tau,
                    use_ema=args.use_ema,
                    dynamic_lr=False,
                    )
    model.fit(X_train[:, None], y_train)
    y_val_pred = model.predict(X_val[:, None])
    kdl = scipy.stats.kendalltau(y_val_pred, y_val).correlation
    print(kdl)