# -*-coding:utf-8-*-
"""
@Time   : 2019-02-22 10:33
@Author : Mark
@File   : rnn_classifier_tf.py
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

TRAIN_DATA_PATH = '../data/train.csv'
TEST_SIZE = 0.2
BATCH_SIZE = 128
N_STEP = 28
N_INPUT = 28
N_HIDDEN_UNITS = 64
N_CLASSES = 10
EPOCH = 100
LR = 0.001


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


data = pd.read_csv(TRAIN_DATA_PATH)

X = data.iloc[:, 1:data.shape[1]]
y = data.iloc[:, 0]

train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(X.values, y.values, test_size=TEST_SIZE, shuffle=False)
train_x = train_x_np.reshape(-1, N_STEP, N_INPUT)
train_y = OneHotEncoder(sparse=False).fit_transform(train_y_np.reshape(-1, 1))
test_x = test_x_np.reshape(-1, N_STEP, N_INPUT)
test_y = OneHotEncoder(sparse=False).fit_transform(test_y_np.reshape(-1, 1))

X = tf.placeholder(tf.float32, shape=[None, N_STEP, N_INPUT])
y = tf.placeholder(tf.int32, shape=[None, N_CLASSES])

rnn_fw = tf.contrib.rnn.LSTMCell(num_units=N_HIDDEN_UNITS)
rnn_bw = tf.contrib.rnn.LSTMCell(num_units=N_HIDDEN_UNITS)

# outputs : [64,28,64] * 2
outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw, cell_bw=rnn_bw,
                                                  inputs=X, dtype=tf.float32)
output = tf.concat([outputs[0], outputs[1]], axis=2)[:, -1, :]
predict_y = tf.contrib.layers.fully_connected(inputs=output, num_outputs=N_CLASSES, activation_fn=None)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=predict_y))
train_step = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCH):
        for step in range(int(train_y_np.shape[0] / BATCH_SIZE)):
            data_generator = batch_generator(train_x, train_y, batch_size=BATCH_SIZE)
            b_x, b_y = next(data_generator)
            train_step.run(feed_dict={X: b_x, y: b_y})

            if step % 100 == 0:
                loss_value = sess.run(fetches=loss, feed_dict={X: train_x, y: train_y})

                predict = sess.run(predict_y, feed_dict={X: test_x, y: test_y})
                accuracy = (np.argmax(predict, axis=1) == np.argmax(test_y, axis=1)).sum() / predict.shape[0]

                print('EPOCH : {0} | STEP : {1} | LOSS : {2} | ACCURACY : {3}'.format(str(epoch),
                                                                                      str(step),
                                                                                      str(loss_value),
                                                                                      str(accuracy)))
                print(np.argmax(predict[0:9, :], axis=1))
                print(np.argmax(test_y[0:9, :], axis=1))
