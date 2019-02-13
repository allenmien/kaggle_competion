# -*-coding:utf-8-*-
"""
@Time   : 2019-02-13 11:08
@Author : Mark
@File   : svm_classifier.py

Kaggle Data: https://www.kaggle.com/c/digit-recognizer/data
"""

print(__doc__)
import os

import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import pca

file_path = "./data/svm_out.csv"
train_pd = pd.read_csv('./data/train.csv')
test_pd = pd.read_csv('./data/test.csv')

train_X_pd = train_pd.iloc[:, 1:]
train_Y_pd = train_pd.iloc[:, :1]
test_X_pd = test_pd.iloc[:, :]

train_X = train_X_pd.values
train_Y = train_Y_pd.values
test_X = test_X_pd.values

pca = pca.PCA(n_components=0.8, whiten=True)
train_X = pca.fit_transform(train_X)
test_X = pca.fit_transform(test_X)
print(pca.explained_variance_ratio_)
print(np.sum(pca.explained_variance_ratio_))

model = svm.SVC(kernel='rbf', C=10)
model.fit(train_X, train_Y)

test_y_predict = model.predict(test_X)

predict_pd = pd.DataFrame({'ImageId': range(1, len(test_X) + 1), 'Label': test_y_predict})

if os.path.exists(file_path):
    os.remove(file_path)
predict_pd.to_csv(file_path, encoding="utf-8", header=True, index=False)
