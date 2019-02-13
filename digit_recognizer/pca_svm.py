# -*-coding:utf-8-*-
"""
@Time   : 2019-02-13 15:48
@Author : Mark
@File   : pca_svm.py
"""
import pandas as pd
from sklearn.decomposition import PCA
from sklearn import svm

# The competition datafiles are in the directory ../input
# Read competition data files:
train = pd.read_csv("./data/train.csv")
test = pd.read_csv("./data/test.csv")

train_x = train.values[:, 1:]
train_y = train.ix[:, 0]
test_x = test.values

pca = PCA(n_components=0.8, whiten=True)
train_x = pca.fit_transform(train_x)
test_x = pca.transform(test_x)

svc = svm.SVC(kernel='rbf', C=10)
svc.fit(train_x, train_y)

test_y = svc.predict(test_x)
pd.DataFrame({"ImageId": range(1, len(test_y) + 1), "Label": test_y}).to_csv('./data/pca_svm_out.csv', index=False, header=True)
