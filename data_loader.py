import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class PHMDataLoader(object):
    def __init__(self, data_path, win_size, step, mode="train"):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        train_df = pd.read_csv(data_path + '/train.txt', sep=" ", header=None)
        train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
        #  print(train_df.shape) (45918,26)

        columns = ["Section-{}".format(i) for i in range(26)]
        train_df.columns = columns

        scaler = MinMaxScaler(feature_range=(0, 1))
        train_norm = scaler.fit_transform(train_df.iloc[:, 2:])
        # print(train_norm.shape) (45918, 24)
        train_df = train_df.iloc[:, :2]
        train_norm = pd.DataFrame(train_norm)
        train_df = pd.concat([train_df, train_norm], axis=1)

        test_df = pd.read_csv(data_path + '/test.txt', sep=" ", header=None)
        test_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)

        test_df.columns = columns

        test_norm = scaler.fit_transform(test_df.iloc[:, 2:])
        # print(train_norm.shape) (45918, 24)
        test_df = test_df.iloc[:, :2]
        test_norm = pd.DataFrame(test_norm)
        test_df = pd.concat([test_df, test_norm], axis=1)
        self.train = train_df.to_numpy()
        self.test = test_df.to_numpy()

    def __len__(self):
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif self.mode == 'test':
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.mode == "train":
            return np.float32(self.train[index:index + self.win_size])
        elif self.mode == "test":
            return np.float32(self.test[index:index + self.win_size])


def get_loader_segment(data_path, batch_size, win_size=20, mode='train'):
    dataset = PHMDataLoader(data_path, win_size, 1, mode)

    shuffle = False
    if mode == 'train':
        shuffle = True
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return data_loader
