# -*-coding:utf-8-*-
"""
@Time   : 2019-02-25 18:08
@Author : Mark
@File   : cnn_classifier_tf.py
"""
import random

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

TEST_SIZE = 0.2
H = 28
W = 28
C = 1
N_CLASSES = 10
LR = 0.01
BATCH_SIZE = 128
EPOCH = 10

train_pd = pd.read_csv('../data/train.csv')
train_x_pd = train_pd.iloc[:, 1:train_pd.shape[1]]
train_y_pd = train_pd.iloc[:, 0]

train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(train_x_pd.values, train_y_pd.values,
                                                                test_size=TEST_SIZE)
train_x_np = train_x_np.reshape(-1, H, W, C)
train_y_np = OneHotEncoder(sparse=False).fit_transform(train_y_np.reshape(-1, 1))
test_x_np = test_x_np.reshape(-1, H, W, C)
test_y_np = OneHotEncoder(sparse=False).fit_transform(test_y_np.reshape(-1, 1))


def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def batch_generator(X, y, batch_size):
    size = X.shape[0]
    X_copy = X.copy()
    y_copy = y.copy()
    indices = np.arange(size)
    np.random.shuffle(indices)
    X_copy = X_copy[indices]
    y_copy = y_copy[indices]

    idx = 0
    while True:
        if idx + batch_size <= size:
            yield X_copy[idx:idx + batch_size], y_copy[idx:idx + batch_size]
            idx += batch_size
        else:
            idx = 0
            indices = np.arange(size)
            np.random.shuffle(indices)
            X_copy = X_copy[indices]
            y_copy = y_copy[indices]
            continue


X = tf.placeholder(tf.float32, shape=(None, H, W, C))
y = tf.placeholder(tf.int32, shape=(None, N_CLASSES))

conv_1 = tf.nn.conv2d(input=X, filter=tf.Variable(tf.random_normal([3, 3, 1, 16])), strides=[1, 1, 1, 1],
                      padding="SAME", data_format="NHWC")
pool_1 = tf.nn.max_pool(value=conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='VALID', data_format='NHWC')
conv_2 = tf.nn.conv2d(input=pool_1, filter=tf.Variable(tf.random_normal([3, 3, 16, 32])), strides=[1, 1, 1, 1],
                      padding="SAME", data_format="NHWC")
pool_2 = tf.nn.max_pool(value=conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                        padding='VALID', data_format='NHWC')

W = init_weights([7 * 7 * 32, N_CLASSES])
b = init_weights([N_CLASSES])

y_pred = tf.matmul(tf.reshape(pool_2, shape=[-1, 7 * 7 * 32]), W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))

train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred, axis=1), tf.argmax(y, axis=1)), dtype=tf.float32))

with tf.Session() as sess:
    generator = batch_generator(X=train_x_np, y=train_y_np, batch_size=BATCH_SIZE)
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH):
        for step in range(int(train_x_np.shape[0] / BATCH_SIZE)):
            batch_x, batch_y = next(generator)
            train_step.run(feed_dict={X: batch_x, y: batch_y})
            _loss = sess.run(loss, feed_dict={X: batch_x, y: batch_y})
            if step % 100 == 0:
                _accuracy = sess.run(accuracy, feed_dict={X: test_x_np, y: test_y_np})
                print('EPOCH : {0} | STEP: {1} | LOSS : {2} | ACCURACY : {3}'.format(str(epoch), str(step), str(_loss),
                                                                                     str(_accuracy)))
                print(sess.run(tf.argmax(y_pred, axis=1), feed_dict={X: test_x_np})[:9])
                print(np.argmax(test_y_np[:9], axis=1))
