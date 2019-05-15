import tensorflow as tf 
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def add_layer(inputs, layer_n, input_size, output_size, activation_function=None):
    layer_name = '{}'.format(layer_n)
    with tf.name_scope('layer{}'.format(layer_n)):
        with tf.name_scope('Weights'):
            Weights = tf.Variable(tf.random_normal([input_size, output_size])) # Variable里面不需要定义数据类型
            tf.summary.histogram(layer_name+'/Weights', Weights)
        with tf.name_scope('Bias'):
            bias = tf.Variable(tf.zeros([1, output_size]) + 0.10) # bias维度为[1,outputsize] wx col为output_size, 可广播
            tf.summary.histogram(layer_name+'/bias', bias)
        with tf.name_scope('Result'):
            wx_plus_bias = tf.add(tf.matmul(inputs, Weights), bias)
        if activation_function == None:
            output = wx_plus_bias
        else:
            output = activation_function(wx_plus_bias)
        tf.summary.histogram(layer_name+'/output', output)    
    return output


X = np.linspace(-1,1, 300)[:, np.newaxis].astype(np.float32)
noise = np.random.normal(0, 0.05, X.shape).astype(np.float32)
y = np.square(X) + noise

xs = tf.placeholder(tf.float32, [None, 1])
ys = tf.placeholder(tf.float32, [None, 1])

l1 = add_layer(xs, 1, 1, 10, tf.nn.relu)
l2 = add_layer(l1, 2, 10, 1)

with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(l2 - y), reduction_indices=[1]))
    tf.summary.scalar('Loss', loss)
with tf.name_scope('Train'):
    train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
    # tf.summary.scalar('Train', train)

init = tf.global_variables_initializer()

with tf.Session() as sess:    
    mergerd = tf.summary.merge_all()
    write = tf.summary.FileWriter('./logs', sess.graph)
    sess.run(init, feed_dict={xs: X, ys: y}) # 运行初始化
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
#     ax.scatter(X, y)
#     plt.ion()
    for i in range(3000):
        sess.run(train, feed_dict={xs: X, ys: y}) # 运行train
        if i % 100 == 0:
            rs = sess.run(mergerd, feed_dict={xs:X, ys: y})
            write.add_summary(rs, i)
#             try:
#                 ax.lines.remove(lines[0]) # 去除掉lines的第一个
#             except Exception:
#                 pass
#             prediction_value = sess.run(l2, feed_dict={xs:X, ys: y})
#             lines = ax.plot(X, prediction_value, 'r-', lw=10)
#             plt.pause(0.2) # 画图暂停0.1s
#     plt.ioff()
#     plt.show()