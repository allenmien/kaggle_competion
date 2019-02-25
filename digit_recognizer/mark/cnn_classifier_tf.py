# -*-coding:utf-8-*-
"""
@Time   : 2019-02-25 18:08
@Author : Mark
@File   : cnn_classifier_tf.py
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

TEST_SIZE = 0.2
H = 28
W = 28
N_CLASSES = 10

train_pd = pd.read_csv('../data/train.csv')
train_x_pd = train_pd.iloc[:, 1:train_pd.shape[1]]
train_y_pd = train_pd.iloc[:, 0]

train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(train_x_pd.values, train_y_pd.values,
                                                                test_size=TEST_SIZE)
train_x_np = train_x_np.reshape(-1, H, W)
train_y_np = OneHotEncoder(sparse=False).fit_transform(train_y_np.reshape(-1, 1))
test_x_np = test_x_np.reshape(-1, H, W)
test_y_np = OneHotEncoder(sparse=False).fit_transform(test_y_np.reshape(-1, 1))

X = tf.placeholder(tf.float32, shape=(None, H, W))
y = tf.placeholder(tf.int32, shape=(None, N_CLASSES))

conv1 = tf.nn.conv2d(input=X, filter=[3, 3, 1, 16], strides=[1], padding="SAME")
pass
