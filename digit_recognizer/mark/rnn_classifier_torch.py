# -*-coding:utf-8-*-
"""
@Time   : 2019-02-21 10:44
@Author : Mark
@File   : rnn_classifier_torch.py
"""
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 50
EPOCH = 3
LR = 0.01

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
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=2, batch_first=True, dropout=0.2,
                            bidirectional=True)
        self.liner = nn.Linear(64 * 2, 10)

    def forward(self, X):
        output, (h_n, c_n) = self.lstm(X, None)
        pred = self.liner(output[:, -1, :])
        return pred


rnn = RNN()

optimizer = torch.optim.Adam(rnn.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        pred = rnn(x.reshape(-1, 28, 28))
        loss = loss_func(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            test_out = rnn.forward(test_x.reshape(-1, 28, 28))
            prediction = torch.argmax(test_out, dim=1)
            accuracy = ((prediction == test_y).sum().float() / test_y.shape[0]).item()
            print('epoch : {0} | step : {1} | loss = {2} | accuracy = {3}'.format(str(epoch),
                                                                                  str(step),
                                                                                  str(loss.item()),
                                                                                  str(accuracy)))
            print(prediction[:10])
            print(test_y.squeeze()[:10])
