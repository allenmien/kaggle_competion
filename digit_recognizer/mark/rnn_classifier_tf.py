# -*-coding:utf-8-*-
"""
@Time   : 2019-02-22 10:33
@Author : Mark
@File   : rnn_classifier_tf.py
"""
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

TRAIN_DATA_PATH = '../data/train.csv'
BATCH_SIZE = 50
LR = 0.01

data = pd.read_csv(TRAIN_DATA_PATH)

X = data.iloc[:, 1:data.shape[1]]
y = data.iloc[:, 1]

train_x_np, test_x_np, train_y_np, test_y_np = train_test_split(X.values, y.values, test_size=0.2, shuffle=False)

input_queue = tf.train.slice_input_producer([train_x_np, train_y_np])

train_x, train_y = tf.train.shuffle_batch([input_queue[0],
                                           input_queue[1]],
                                          batch_size=BATCH_SIZE,
                                          capacity=100 + BATCH_SIZE * 3,
                                          min_after_dequeue=50,
                                          num_threads=4)

train_x = tf.reshape(tf.to_float(train_x), (-1, 28, 28))
train_y = tf.reshape(tf.to_float(train_y), (-1, 1))

rnn_fw = tf.contrib.rnn.LSTMCell(num_units=64)
rnn_bw = tf.contrib.rnn.LSTMCell(num_units=64)

with tf.Session() as session:
    # outputs : [50,28,64] * 2
    outputs, states = tf.nn.bidirectional_dynamic_rnn(cell_fw=rnn_fw, cell_bw=rnn_bw,
                                                      inputs=train_x, dtype=tf.float32)
    output = tf.concat([outputs[0], outputs[1]], axis=2)[:, -1, :]
    predict_y = tf.contrib.layers.fully_connected(inputs=output, num_outputs=10, activation_fn=tf.nn.relu)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=train_y, logits=predict_y)
    optimizer = tf.train.AdamOptimizer(learning_rate=LR).minimize(loss)
    print(optimizer)
