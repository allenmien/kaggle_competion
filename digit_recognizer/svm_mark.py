# -*-coding:utf-8-*-
"""
@Time   : 2019-02-13 11:08
@Author : Mark
@File   : svm_mark.py

Kaggle Data: https://www.kaggle.com/c/digit-recognizer/data
"""

print(__doc__)
import os

import pandas as pd
from sklearn import svm

file_path = "./data/svm_out.csv"
train_pd = pd.read_csv('./data/train.csv')
test_pd = pd.read_csv('./data/test.csv')

train_X_pd = train_pd.iloc[:50, 1:]
train_Y_pd = train_pd.iloc[:50, :1]
test_X_pd = test_pd.iloc[:50, :]

train_X = train_X_pd.values
train_Y = train_Y_pd.values
test_X = test_X_pd.values

clf = svm.SVC()
clf.fit(train_X, train_Y)

test_y_predict = clf.predict(test_X)

predict_pd = pd.DataFrame({'ImageId': range(1, len(test_X) + 1), 'Label': test_y_predict})

if os.path.exists(file_path):
    os.remove(file_path)
predict_pd.to_csv(file_path, encoding="utf-8", header=True, index=False)
