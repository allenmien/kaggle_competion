# -*-coding:utf-8-*-
"""
@Time   : 2019-02-14 10:13
@Author : Mark
@File   : cnn_classifier_torch.py
"""
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cross_validation import train_test_split
from torch.utils.data import TensorDataset, DataLoader

BATCH_SIZE = 50
EPOCH = 3
LR = 0.001


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1, dilation=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU()
        )
        self.liner = nn.Linear(in_features=7 * 7 * 32, out_features=10)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        out = self.liner(x.reshape(x.shape[0], -1))
        return out


train_pd = pd.read_csv('../data/train.csv')

train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(train_pd.values[:, 1:],
                                                                train_pd.values[:, :1],
                                                                test_size=0.2)

train_X = torch.FloatTensor(train_x_np)
train_Y = torch.LongTensor(train_y_np)
test_X = torch.FloatTensor(test_x_np)
test_Y = torch.LongTensor(test_y_np)

deal_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(dataset=deal_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Model()
# print(model)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(train_loader):
        out = model.forward(x.reshape(BATCH_SIZE, 1, 28, 28))
        loss = loss_func(out, y.reshape(BATCH_SIZE))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            test_out = model.forward(test_X.reshape(-1, 1, 28, 28))
            prediction = torch.argmax(test_out, dim=1)
            correct = (prediction == test_Y.squeeze()).sum().float()
            accuracy = (correct / (len(test_Y))).item()
            print('epoch : {0} | step : {1} | loss = {2} | accuracy = {3}'.format(str(epoch),
                                                                                  str(step),
                                                                                  str(loss.item()),
                                                                                  str(accuracy)))
            print(prediction[:10])
            print(test_Y.squeeze()[:10])
torch.save(model, '../model/cnn_classifier_torch.pkl')
