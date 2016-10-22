import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math


def main():
    # Define the computation graph
    X = tf.placeholder(tf.float32, [None])
    Y_ = tf.placeholder(tf.float32, [None])
    a = tf.Variable(0.0)
    b = tf.Variable(0.0)

    # Linear regression
    Y = a * X + b

    # Quadratic loss
    loss = (Y-Y_)**2

    # Gradients
    grad_a = 2 * a * (a*X + b - Y_)
    grad_b = 2 * (a*X + b - Y_)

    # core_inv = tf.matrix_inverse(tf.matmul(X, X, transpose_a=True))
    # pseudo_inverz = tf.matmul(core_inv, X, transpose_b=True)
    # grad_w = tf.matmul(pseudo_inverz, Y_)

    # Optimization via gradient descent
    direct = False
    trainer = tf.train.GradientDescentOptimizer(0.1)
    if direct:
        train_op = trainer.minimize(loss)
    else:
        grads_and_vars = trainer.compute_gradients(loss, [a, b])
        train_op = trainer.apply_gradients(grads_and_vars)

    # Initialise the TF parameters
    sess = tf.Session()
    sess.run(tf.initialize_all_variables())

    # Inputs
    feed_dict = {X: [1, 2], Y_: [3, 5]}

    # Train the model
    for i in range(100):
        val_loss, _, val_a, val_b = sess.run([loss, train_op, a, b], feed_dict=feed_dict)
        print(i, val_loss,  val_a, val_b)

        grads = sess.run(grads_and_vars, feed_dict=feed_dict)
        print('Grads[trainer]:', [g[0] for g in grads])

        custom_grad_a, custom_grad_b = sess.run([grad_a, grad_b], feed_dict=feed_dict)
        print('Gradient a:', custom_grad_a)
        print('Gradient b:', custom_grad_b)

        a = tf.Print(custom_grad_a, [custom_grad_a], message="Gradient of a: ")
        b = tf.Print(custom_grad_b, [custom_grad_b], message="Gradient of b: ")
        print()


if __name__ == '__main__':
    main()