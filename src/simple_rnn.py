"""
Very simple demo RNNs.
"""

import argparse
import numpy as np
import tensorflow as tf

from tensorflow.contrib.layers import fully_connected


def do_mnist():
    from tensorflow.examples.tutorials.mnist import input_data

    n_steps = 28
    n_inputs = 28
    n_neurons = 150
    n_outputs = 10

    learning_rate = 0.001

    X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.int32, [None])

    basic_cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons)
    outputs,states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

    logits = fully_connected(states, n_outputs, activation_fn=None)
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
            logits=logits)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

    init = tf.global_variables_initializer()

    mnist = input_data.read_data_sets("./data")
    X_test = mnist.test.images.reshape((-1, n_steps, n_inputs))
    y_test = mnist.test.labels

    n_epochs = 100
    batch_size = 150

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(mnist.train.num_examples // batch_size):
                X_batch,y_batch = mnist.train.next_batch(batch_size)
                X_batch = X_batch.reshape((-1, n_steps, n_inputs))
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy.eval( feed_dict={X: X_batch, y: y_batch} )
            acc_test = accuracy.eval( feed_dict={X: X_test, y: y_test} )
            print("%d: Train accuracy: %f Test accuracy: %f" \
                    % (epoch, acc_train, acc_test))

def predict_sin():
    N = 300
    t = np.linspace(0, 30, N)
    f_t = t*np.sin(t)/3 + 2*np.sin(5*t)

    n_steps = 20
    n_inputs = 1 
    n_neurons = 100
    n_outputs = 1

    learning_rate = 0.001

    x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
    y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])

    cell = tf.contrib.rnn.OutputProjectionWrapper(
            tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, 
                activation=tf.nn.relu),
            output_size=n_outputs)
    outputs,states = tf.nn.dynamic_rnn(cell, x, dtype=tf.float32)

    loss = tf.reduce_mean( tf.square(outputs - y) )
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    training_op = optimizer.minimize(loss)

    init = tf.global_variables_initializer()

    n_epochs = 10000
    batch_size = 50

    def _get_batch(iteration):
        X_batch = []
        y_batch = []
        for i in range(batch_size):
            idx_bgn = iteration*batch_size + i
            idx_end = idx_bgn + n_steps
            if idx_end == N:
                break
            X_batch.append( t[ idx_bgn : idx_end ] )
            y_batch.append( f_t[ idx_bgn : idx_end ] )
        X_batch = np.array(X_batch)
        y_batch = np.array(y_batch)
        X_batch = X_batch.reshape((-1, n_steps, n_inputs))
        y_batch = y_batch.reshape((-1, n_steps, n_outputs))
        return X_batch,y_batch 

    with tf.Session() as sess:
        init.run()
        for epoch in range(n_epochs):
            for iteration in range(N // batch_size):
                X_batch,y_batch = _get_batch(iteration) 
                sess.run(training_op, feed_dict={x: X_batch, y: y_batch})
            if epoch % 100 == 0:
                mse = loss.eval(session=sess, 
                        feed_dict={x: X_batch, y: y_batch})
                print("Epoch %d, mse: %f" % (epoch, mse))



def main(args):
#    do_mnist()
    predict_sin()


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)
