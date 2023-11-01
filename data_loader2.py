import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

data_path = "F:\Challenge_Data"
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

win_size = 20


def gen_sequence(id_df, win_size, features):
    data_array = id_df[features].values
    num_elements = data_array.shape[0]
    for start, stop in zip(range(0, num_elements - win_size), range(win_size, num_elements)):
        yield data_array[start:stop, :]


features = [i for i in range(0, 24)]
seq_gen = (list(gen_sequence(train_df[train_df['Section-0'] == id], win_size, features))
           for id in train_df["Section-0"].unique())
seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
# print(seq_array)
# print(seq_array.shape) (41558, 20, 24)

seq_gen_test = [test_df[test_df['Section-0'] == id][features].values[-win_size:]
           for id in test_df["Section-0"].unique() if len(test_df[test_df['Section-0']==id]) >= win_size]
seq_array_test = np.asarray(seq_gen_test).astype(np.float32)
print(seq_array_test.shape)


train = seq_array
test = seq_array_test


def get_loader_segment(batch_size, win_size=20, step=1, mode='tarin'):
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
