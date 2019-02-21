# -*-coding:utf-8-*-
"""
@Time   : 2019-02-21 10:44
@Author : Mark
@File   : rnn_classifier_torch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split

BATCH_SIZE = 50
EPOCH = 3
LR = 0.001
COUNT = 0

data = pd.read_csv('../data/train.csv')

X = data.iloc[:, 1:data.shape[1]].values
y = data.iloc[:, 0].values

train_x_array, test_x_array, train_y_array, test_y_array = train_test_split(X, y, test_size=0.2, shuffle=True)

train_x = torch.FloatTensor(train_x_array)
train_y = torch.LongTensor(train_y_array)
test_x = torch.FloatTensor(test_x_array)
test_y = torch.LongTensor(test_y_array)

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, num_layers=2)
        self.liner = nn.Linear(128, 10)

    def forward(self, X):
        output, (h_n, c_n) = self.lstm(X)
        pred = self.liner(output[-1, :, :])
        return pred


rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        pred = rnn(x.reshape(28, BATCH_SIZE, 28))
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        COUNT += 1

        if (COUNT - 1) % 10 == 0:
            pred_test_y = rnn(test_x.reshape(28, -1, 28))
            accuracy = (torch.argmax(pred_test_y) == test_y).sum().float() / test_x.shape[0]
            print('Count : {0} | EPOCH : {1} | STEP : {2} | Loss : {3} | accuracy : {4}'.format(str(COUNT), str(epoch),
                                                                                                str(step),
                                                                                                str(loss.item()),
                                                                                                str(accuracy.item())))
