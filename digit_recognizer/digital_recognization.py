# -*-coding:utf-8-*-
"""
@Time   : 2019-02-13 11:08
@Author : Mark
@File   : digital_recognization.py

Kaggle Data: https://www.kaggle.com/c/digit-recognizer/data
"""

print(__doc__)
import pandas as pd
from sklearn import svm
import numpy as np

train_pd = pd.read_csv('./data/train.csv')
test_pd = pd.read_csv('./data/test.csv')

train_X_pd = train_pd.iloc[:, 1:]
train_Y_pd = train_pd.iloc[:, :1]
test_X_pd = test_pd.iloc[:, 1:]
test_Y_pd = test_pd.iloc[:, :1]

train_X = train_X_pd.values
train_Y = train_Y_pd.values
test_X = test_X_pd.values
test_Y = test_Y_pd.values

clf = svm.SVC()
clf.fit(train_X, train_Y)

p = clf.get_params()
test_y_predict = clf.predict(test_X)
P = clf.score()
pass
