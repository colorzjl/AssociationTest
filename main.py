import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pandas as pd
from model import AnomalyTransformer, s, p
# from data_loader2 import train, test, get_loader_segment
from data_loder_switch import train, test, get_loader_segment
import matplotlib.pyplot as plt
import seaborn as sns


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name='', delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model, path):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
        elif score < self.best_score + self.delta or score2 < self.best_score2 + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), os.path.join(path, str(self.dataset) + '_checkpoint.pth'))
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = nn.MSELoss()
model = AnomalyTransformer(win_size=100, enc_in=20, c_out=20, e_layers=3)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
if torch.cuda.is_available():
    model.cuda()

train_loader = get_loader_segment(batch_size=128, mode='train')


def train():
    '''
    训练
    :return:
    '''
    win_size = 100
    k = 3
    time_now = time.time()
    early_stopping = EarlyStopping(patience=3, verbose=True, dataset_name=train)
    train_steps = len(train_loader)
    num_epochs = 3
    loss1_list = []
    df = pd.DataFrame(columns=['Epoch', 'train_loss'])
    for epoch in range(num_epochs):
        iter_count = 0
        epoch_time = time.time()
        model.train()
        for i, input_data in enumerate(train_loader):
            # print("在跑了在跑了")
            optimizer.zero_grad()
            iter_count += 1
            input = input_data.float().to(device)

            output, series, prior, _ = model(input)
            # calculate Association discrepancy
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += (torch.mean(my_kl_loss(series[u], (
                        prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                               win_size)).detach())) + torch.mean(
                    my_kl_loss((prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                       win_size)).detach(),
                               series[u])))
                prior_loss += (torch.mean(my_kl_loss(
                    (prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                            win_size)),
                    series[u].detach())) + torch.mean(
                    my_kl_loss(series[u].detach(), (
                            prior[u] / torch.unsqueeze(torch.sum(prior[u], dim=-1), dim=-1).repeat(1, 1, 1,
                                                                                                   win_size)))))
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = criterion(output, input)
            loss1_list.append((rec_loss - k * series_loss).item())
            loss1 = rec_loss - k * series_loss
            loss2 = rec_loss + k * prior_loss
            if (i + 1) % 100 == 0:
                speed = (time.time() - time_now) / iter_count
                left_time = speed * ((num_epochs - epoch) * train_steps - i)
                print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                iter_count = 0
                time_now = time.time()
            loss1.backward(retain_graph=True)
            loss2.backward()
            optimizer.step()
        print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
        train_loss = np.average(loss1_list)
        print(
            "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} ".format(
                epoch + 1, train_steps, train_loss))

        adjust_learning_rate(optimizer, epoch + 1, lr_=0.001)

    def draw_loss(Loss_list, epoch):
        # 我这里迭代了200次，所以x的取值范围为(0，200)，然后再将每次相对应的准确率以及损失率附在x上
        plt.cla()
        x1 = range(1, epoch * train_steps + 1)
        # print(x1)
        y1 = Loss_list
        # print(y1)
        plt.title('Train loss', fontsize=20)
        plt.plot(x1, y1, '.-')
        plt.xlabel('steps', fontsize=20)
        plt.ylabel('Train loss', fontsize=20)
        plt.grid()
        plt.show()

    draw_loss(loss1_list, num_epochs)


train()
# 保存参数
# torch.save(model.state_dict(), 'params_win_50.pth')
print("===================================================================")

# print("s.len:{}".format(len(s)))
# print("pri.len:{}".format(len(p)))
# print("s[0].shape:{}".format(s[0].shape))
# print("p[0].shape:{}".format(p[0].shape))
# print(s[1])
sns.set(style='whitegrid', color_codes=True)
t = np.mean(s, axis=0)
# 不用再转了，他本来就是array
print("t.type:{}".format(type(t)))
print("t.shape:{}".format(t.shape))
ax = sns.heatmap(t,cmap="Greens")
plt.plot()
plt.show()

# print("pri.len:{}".format(len(p)))
# print("shape of pri:{}".format(p[-1].shape))
sns.set(style='whitegrid', color_codes=True)
pri = np.mean(p, axis=0)
ax = sns.heatmap(pri,cmap="Greens")
plt.plot()
plt.show()
