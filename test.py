import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import AnomalyTransformer,s
from data_loader2 import get_loader_segment, test
import torch.nn as nn
import seaborn as sns

model = AnomalyTransformer(win_size=50, enc_in=24, c_out=24, e_layers=3)
test_loader = get_loader_segment(batch_size=128, mode='test')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()


def test():

    model.load_state_dict(torch.load('params_win_50.pth'))
    model.eval()
    loss_list = []
    with torch.no_grad():
        for i, input_data in enumerate(test_loader):
            input = input_data.float().to(device)
            output, series, prior, _ = model(input)
            loss = criterion(input, output)
            loss_list.append(loss)
        MSE = np.average(loss_list)
        print("test_MSE:",MSE)


test()

print("s.len:{}".format(len(s)))
print("s[1].shape:{}".format(s[-1].shape))
print(s[1])
sns.set(style='whitegrid', color_codes=True)
t = s[1]
t1 = s[10]
t2 = s[50]
t3 = s[100]
# 不用再转了，他本来就是array
print("t.tyepe:{}".format(type(t)))


ax = sns.heatmap(t)
plt.show()
ax1 = sns.heatmap(t1)
plt.show()
ax2 = sns.heatmap(t2)
plt.show()
ax3 = sns.heatmap(t3)
plt.show()
