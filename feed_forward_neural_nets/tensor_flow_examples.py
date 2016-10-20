import tensorflow as tf
import numpy as np


def main():
    example_3()


def example_1():
    # form the computational graph
    a = tf.constant(5)
    b = tf.constant(8)
    x = tf.placeholder(dtype='int32')
    c = a + b * x
    d = b * x

    # create query phase and run context
    session = tf.Session()

    # query: calculate c with x=5
    c_val = session.run(c, feed_dict={x: 5})

    # print result
    print(c_val)


def example_2():
    X = tf.placeholder(tf.float32, [2, 2])
    Y = 3 * X + 5
    z = Y[0, 0]
    sess = tf.Session()

    Y_val = sess.run(Y, feed_dict={X: [[0, 1], [2, 3]]})
    z_val = sess.run(z, feed_dict={X: np.ones([2, 2])})

    print(Y_val[0,0], type(Y_val))
    print(z_val, type(z_val))


def example_3():
    X = tf.placeholder(tf.float32, [None, 5])
    Y = 3 * X + 5

    sess = tf.Session()
    Y_val = sess.run(Y, feed_dict={X: np.ones((3,5))})

    print(Y.get_shape())
    print(sess.run(tf.shape(Y), feed_dict={X: np.ones((3,5))}))


if __name__ == '__main__':
    main()
