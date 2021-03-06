import tensorflow as tf 
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)

# data 
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# set hyperparameters
learn_rate = 0.001
training_iters = 100000
batch_size = 128
n_inputs = 28 # MNIST data input (image shape 28 * 28)
n_steps = 28 # rnn cells numbers 一张图片每次输入28个pixes，需输入28次，按照时间顺序需要查看28步
n_hidden_layers = 128 # 进入cells之前的隐藏层
n_classes = 10 # cells输出之后的隐藏层，得到结果

# tf Graph Input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# define weights
weights = {
    'in': tf.Variable(tf.truncated_normal([n_inputs, n_hidden_layers])), # 28 * 128
    'out': tf.Variable(tf.truncated_normal([n_hidden_layers, n_classes])) # 128 * 10
}

bias = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_layers])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

def RNN(X, weights, bias):
    # hidden layer for input to cell
    ################################

    # transpose the inputs shape from x --> (128batch * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # into hidden
    X_in = tf.matmul(X, weights['in']) + bias['in'] # 矩阵乘法只能2维*2维，所以要先transpose，然后转回去

    # retranspose
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_layers])



    # cell
    ################################
    cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_layers)
    init_state = cell.zero_state(batch_size, dtype=tf.float32) # 每一批的每一个是独立的

    outputs, final_state = tf.nn.dynamic_rnn(cell, X_in, initial_state=init_state, time_major=False)






    # hidden layer for output as the final results
    # method1: 
    # result = tf.matmul(final_state[1], weights['out']) + bias['out'] # final_state:(major state, side state)
    # method2:
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], weights['out']) + bias['out']

    return results

pred = RNN(x, weights, bias)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))  

train_step = tf.train.AdamOptimizer(learn_rate).minimize(loss)

accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1)), dtype=tf.float32))

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    times = 0 
    while times*batch_size < training_iters:
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        test_x, test_y = mnist.test.next_batch(batch_size)
        batch_x = batch_x.reshape([batch_size, n_steps, n_inputs])
        test_x = test_x.reshape([batch_size, n_steps, n_inputs])
        sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        if times % 20 == 0:
            print(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))
        times+=1
