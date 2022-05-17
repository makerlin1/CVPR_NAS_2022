import json
from torch.utils.data import Dataset
import numpy as np
from dataset import convert_X
RANK_NAME = ['cplfw_rank', 'market1501_rank', 'dukemtmc_rank', 'msmt17_rank', 'veri_rank', 'vehicleid_rank',
             'veriwild_rank', 'sop_rank']


class NASDataset(Dataset):
    def __init__(self, dataset_path='../data/CVPR_2022_NAS_Track2_train.json', use_ranks=None, transform=None,
                 transform_label=None):
        super(NASDataset, self).__init__()
        with open(dataset_path, 'r') as f:
            train_data = json.load(f)
        self.data_num = len(train_data.keys())

        train_list = []
        arch_list_train = []
        if transform is None:
            transform = self._convert_x
        if transform_label is None:
            transform_label = self._convert_y
        self.rank_names = RANK_NAME
        if use_ranks is not None:
            self.rank_names = use_ranks
        for key in train_data.keys():
            tmp_list = []
            for idx, name in enumerate(self.rank_names):
                tmp_list.append(train_data[key][name])
            train_list.append(transform_label(tmp_list))
            arch_list_train.append(transform(train_data[key]['arch']))
        self.arch_list_train = arch_list_train
        self.train_list = train_list

    def __getitem__(self, index):
        return self.arch_list_train[index], self.train_list[index]

    def __len__(self):
        return self.data_num

    def _convert_x(self, arch_str):
        temp_arch = []
        for elm in arch_str:
            if elm == 'l':
                temp_arch.append(1 / 3)
            elif elm == 'j':
                temp_arch.append(2 / 3)
            elif elm == 'k':
                temp_arch.append(3 / 3)
            else:
                temp_arch.append(int(elm) / 3)
        return np.array(temp_arch).astype(np.float32)

    def _convert_y(self, x):
        return np.array(x).astype(np.int64)


class NASDatasetV2(Dataset):
    def __init__(self, dataset_path='../data/CVPR_2022_NAS_Track2_train.json', use_ranks=None, transform=None,
                 transform_label=None, mode='trainval', k_fold=0):
        super(NASDatasetV2, self).__init__()
        with open(dataset_path, 'r') as f:
            train_data = json.load(f)
        train_list = []
        arch_list_train = []
        if transform is None:
            transform = self._convert_x
        if transform_label is None:
            transform_label = self._convert_y
        self.rank_names = RANK_NAME
        if use_ranks is not None:
            self.rank_names = use_ranks
        for key in train_data.keys():
            tmp_list = []
            for idx, name in enumerate(self.rank_names):
                tmp_list.append(train_data[key][name])
            train_list.append(transform_label(tmp_list))
            arch_list_train.append(transform(train_data[key]['arch']))
        self.arch_list_train = []
        self.train_list = []
        if len(use_ranks) == 1:
            arch_list_train, train_list = self._sort(arch_list_train, train_list)
            if mode == 'trainval':
                self.arch_list_train = [data for idx, data in enumerate(arch_list_train) if (idx + k_fold) % 5 == 0]
                train_list = train_list
            if mode == 'train':
                self.arch_list_train = [data for idx, data in enumerate(arch_list_train) if (idx + k_fold) % 5 != 0]
                self.train_list = [data for idx, data in enumerate(train_list) if (idx + k_fold) % 5 != 0]
            if mode == 'val':
                self.arch_list_train = [data for idx, data in enumerate(arch_list_train) if (idx + k_fold) % 5 == 0]
                self.train_list = [data for idx, data in enumerate(train_list) if (idx + k_fold) % 5 == 0]
        else:
            self.arch_list_train = arch_list_train
            self.train_list = train_list
        self.data_num = len(self.arch_list_train)

    def __getitem__(self, index):
        return np.array(self.arch_list_train[index]).astype(np.int64), np.array(self.train_list[index]).astype(np.float32)

    def __len__(self):
        return self.data_num

    def _convert_x(self, arch_str):
        temp_arch = []
        for elm in arch_str:
            if elm == 'l':
                temp_arch.append(1 / 3)
            elif elm == 'j':
                temp_arch.append(2 / 3)
            elif elm == 'k':
                temp_arch.append(3 / 3)
            else:
                temp_arch.append(int(elm) / 3)
        return np.array(temp_arch, dtype='int64')

    def _convert_y(self, x):
        return x

    def _sort(self, x, y):
        # 遍历所有数组元素
        data_num = len(y)
        for i in range(data_num):
            # Last i elements are already in place
            for j in range(0, data_num - i - 1):
                if y[j][0] > y[j + 1][0]:
                    y[j], y[j + 1] = y[j + 1], y[j]
                    x[j], x[j + 1] = x[j + 1], x[j]
        return x, y


if __name__ == '__main__':
    # 测试定义的数据集
    import torch
    nas_dataset = NASDatasetV2(use_ranks=[RANK_NAME[0]], mode='val', transform=convert_X)
    # print('=============custom dataset=============')
    # for data, label in nas_dataset:
    #     print(data, label)
    #     break
    # train_loader = paddle.io.DataLoader(nas_dataset, batch_size=16, shuffle=True)
    # # 如果要加载内置数据集，将 custom_dataset 换为 train_dataset 即可
    # for batch_id, data in enumerate(train_loader()):
    #     x_data = data[0]
    #     y_data = data[1]
    #     print(x_data)
    #     print(y_data)
    #     break

    #
    loader = torch.utils.data.DataLoader(nas_dataset,
                                         batch_size=5,
                                         shuffle=False,
                                         num_workers=3,
                                         pin_memory=True)
    for (x, y) in loader:
        print(x.shape, y.shape)
        break


    # # # 数据集
    # dataset_size = len(nas_dataset)
    # train_size = int(0.8 * dataset_size)  # 数据集划分
    # test_size = dataset_size - train_size
    # train_dataset, test_dataset = paddle.io.random_split(nas_dataset, [train_size, test_size])
    # print(len(train_dataset))
    # print(len(test_dataset))
