import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data


def main():
    np.random.seed(100)
    tf.set_random_seed(100)

    # Init the dataset
    X, Y_, Yoh_ = data.sample_gmm_2d(K=6, C=2, N=20)

    # Construct the computing graph
    tflr = TFLogreg(X.shape[1], Yoh_.shape[1], param_delta=0.5)

    tflr.train(X, Yoh_, 1000)
    tflr.eval(X, Yoh_)

    # Plot the results
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(tflogreg_classify(X, tflr), bbox, offset=0)
    data.graph_data(X, Y_, tflr.predict(X))

    # show the results
    #plt.savefig('tf_logreg_classification.png')
    plt.show()


def tflogreg_classify(X, model):
    def classify(X):
        return model.predict(X)
    return classify


class TFLogreg(object):

    def __init__(self, D, C, param_delta, param_lambda=0.02):
        self.dimension_num = D
        self.class_num = C

        self.X = tf.placeholder(tf.float32, [None, self.dimension_num])
        self.Yoh_ = tf.placeholder(tf.float32, [None, self.class_num])

        self.W = tf.Variable(tf.zeros([self.dimension_num, self.class_num]))
        self.b = tf.Variable(tf.zeros([self.class_num]))

        self.probs = tf.nn.softmax(tf.matmul(self.X, self.W) + self.b)

        empirical_loss = tf.reduce_mean(-tf.reduce_sum(self.Yoh_ * tf.log(self.probs), reduction_indices=1))
        regularization = param_lambda * tf.nn.l2_loss(self.W)
        self.loss = empirical_loss + regularization

        self.optimizer = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

        self.sess = tf.Session()
        #self.sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

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

    def predict(self, X):
        probs = self.sess.run(self.probs, feed_dict={self.X: X})
        return np.argmax(probs, axis=1)


if __name__ == '__main__':
    main()
