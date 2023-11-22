import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from utils import *

data_path = "./dataset"

train_file_list = getfilelist(data_path + "/train_featuredata")
train_file_list.sort(key=lambda x: int(x.split('_')[-1][:-5]))


# print(file_list) 每一个列表是一个设备全寿命数据,存的是路径
# print(len(file_list)) 27

class MyDataset(Dataset):
    def __init__(self, features, window_size=100):
        self.window_size = window_size
        self.features = features

    def __getitem__(self, index):
        feature = np.float32(self.features[index:index + self.window_size, :])
        return feature

    def __len__(self):
        return len(self.features) - self.window_size


def gen_sequence(file, win_size=100):
    # 这是只有一个序列
    train_df_sub = pd.read_excel(io=file, header=1)
    train_df_sub.columns = train_df_sub.columns.astype(str)
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_df_sub_norm = scaler.fit_transform(train_df_sub)
    length = train_df_sub.shape[0]
    if length >= win_size:
        for start, stop in zip(range(0, length - win_size), range(win_size, length)):
            yield train_df_sub_norm[start:stop,:]
    else:
        yield []


data = list(gen_sequence(file) for file in train_file_list)
print(len(list(data[0])))
data_array = np.concatenate(data).astype(np.float32)
print(data_array.shape)


def get_loader_segment(batch_size, win_size=100, step=1, mode='tarin'):
    shuffle = False
    if mode == 'train':
        dataset = train
        shuffle = True
    elif mode == 'test':
        dataset = test

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    return data_loader