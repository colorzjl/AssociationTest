import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_path = "./data"
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
test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)

test_df.columns = columns

test_norm = scaler.fit_transform(test_df.iloc[:, 2:])

test_df = test_df.iloc[:, :2]
test_norm = pd.DataFrame(test_norm)
test_df = pd.concat([test_df, test_norm], axis=1)


# print(test_df.shape) 29820,26

win_size = 50


# 处理数据
def data_select(data_df, win_size):
    for id in data_df["Section-0"].unique():
        d = data_df[data_df['Section-0'] == id]
        num = d.shape[0]
        if num <= win_size:
            data_df = data_df[data_df['Section-0'] != id]
            # data_df.drop('Section-0' = id)
    return data_df


train_df = data_select(train_df, 50)
test_df = data_select(test_df, 50)


# train_df[train_df['Section-0'] == 1] 就是取section-0=1的数据

def gen_sequence(id_df, win_size, features):
    '''
    :param id_df: 该id对应的数据
    :param win_size: 窗口大小
    :param features: 特征
    :return: 生成最后的数据集，每次生成一个序号对应的数据。然后利用np.concatenate拼在一起
    data_array是取该序号对应数据的特征值
    返回滑动窗口处理好的数据
    '''
    data_array = id_df[features].values
    num_elements = data_array.shape[0]
    if num_elements >= win_size:
        for start, stop in zip(range(0, num_elements - win_size), range(win_size, num_elements)):
            yield data_array[start:stop, :]
    else:
        yield []


features = [i for i in range(0, 24)]
seq_gen = (list(gen_sequence(train_df[train_df['Section-0'] == id], win_size, features))
           for id in train_df["Section-0"].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
# print("train_data_type:",type(seq_array)) np.array
# print(seq_array.shape) (41558, 20, 24)

seq_gen_test = (list(gen_sequence(test_df[test_df['Section-0'] == id], win_size, features))
                for id in test_df["Section-0"].unique())
seq_array_test = np.concatenate(list(seq_gen_test)).astype(np.float32)

train = seq_array
# print(train.shape)
test = seq_array_test
# print(test.shape)


def get_loader_segment(batch_size, win_size=50, step=1, mode='tarin'):
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
