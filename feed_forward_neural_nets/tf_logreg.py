import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
import data


def main():
    np.random.seed(100)
    tf.set_random_seed(100)

    # Init the dataset
    X, Yoh_ = data.sample_gmm_2d(K=6, C=2, N=10, one_hot=True)

    # Construct the computing graph
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], 0.5)

    tflr.train(X, Yoh_, 1000)
    probs = tflr.eval(X, Yoh_)

    print(probs)


class TFLogreg(object):

    def __init__(self, D, C, param_delta=0.05):
        self.dimension_num = D
        self.class_num = C

        self.X = tf.placeholder(tf.float32, [None, self.dimension_num])
        self.Yoh_ = tf.placeholder(tf.float32, [None, self.class_num])

        self.W = tf.Variable(tf.zeros([self.dimension_num, self.class_num]))
        self.b = tf.Variable(tf.zeros([self.class_num]))

        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs), reduction_indices=1))
        self.optimizer = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

        self.sess = tf.Session()

    def train(self, X, Yoh_, param_niter):
        init = tf.initialize_all_variables()
        self.sess.run(init)

        for iteration in range(param_niter):

            _, loss, probs = self.sess.run(
                [self.optimizer, self.loss, self.probs],
                feed_dict={self.X: X, self.Yoh_: Yoh_}
            )

            print(iteration, loss)

    def eval(self, X, Yoh_):
        correct_prediction = tf.equal(tf.argmax(self.probs, 1), tf.argmax(self.Yoh_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({self.X: X, self.Yoh_: Yoh_}, session=self.sess))
        return self.probs


if __name__ == '__main__':
    main()
