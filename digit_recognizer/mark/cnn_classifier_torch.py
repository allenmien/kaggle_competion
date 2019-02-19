# -*-coding:utf-8-*-
"""
@Time   : 2019-02-14 10:13
@Author : Mark
@File   : cnn_classifier_torch.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, TensorDataset, DataLoader
import pandas as pd

BATCH_SIZE = 50
EPOCH = 3


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2, padding=0, dilation=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.liner1 = nn.Linear(in_features=7, out_features=10)

    def forward(self, x):
        conv1_out = self.conv1(x)
        relu1_out = F.relu(conv1_out)
        pool1_out = self.pool1(relu1_out)
        conv2_out = self.conv1(pool1_out)
        relu2_out = F.relu(conv2_out)
        pool2_out = self.pool1(relu2_out)
        out = self.liner1(pool2_out)
        return out


train_pd = pd.read_csv('../data/train.csv')
test_pd = pd.read_csv('../data/test.csv')

train_X = torch.tensor(train_pd.values[:, 1:], requires_grad=False, dtype=torch.float64)
train_Y = torch.tensor(train_pd.values[:, :1], requires_grad=False, dtype=torch.float64)
test_X = torch.tensor(test_pd.values[:, :1], requires_grad=False, dtype=torch.float64)

deal_dataset = TensorDataset(train_X, train_Y)
train_loader = DataLoader(dataset=deal_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = Model()
# print(model)
optimizer = torch.optim.Adam(model.parameters())
loss_func = nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    for step, (x, y) in enumerate(deal_dataset):
        out = model.forward(x.resize_(BATCH_SIZE, 1, 28, 28))
        loss = loss_func(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
