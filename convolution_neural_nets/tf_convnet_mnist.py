import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string('data_dir', '/tmp/data/', 'Directory for storing data')


def main():
    dataset = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
    train_x, train_y, valid_x, valid_y, test_x, test_y = create_dataset(dataset)

    feature_number = dataset.train.images.shape[1]
    class_num = dataset.train.labels.shape[1]

    conv_net = TFConvNet(feature_number, class_num, False)
    conv_net.train(train_x, train_y, test_x, test_y)


def create_dataset(dataset):
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)

    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 28, 28, 1])
    train_y = dataset.train.labels

    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 28, 28, 1])
    valid_y = dataset.validation.labels

    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 28, 28, 1])
    test_y = dataset.test.labels

    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    return train_x, train_y, valid_x, valid_y, test_x, test_y


class TFConvNet(object):
    def __init__(self, feature_num, class_num, is_training, step=0.001):
        self.weight_decay = 1e-3
        self.bn_params = {
            # Decay for the moving averages.
            'decay': 0.999,
            'center': True,
            'scale': True,
            # epsilon to prevent 0s in variance.
            'epsilon': 0.001,
            # None to force the updates during train_op
            'updates_collections': None,
            'is_training': is_training
        }

        self.feature_num = feature_num
        self.class_num = class_num

        self.X = tf.placeholder(tf.float32, [None, feature_num])
        self.y_ = tf.placeholder(tf.float32, [None, class_num])

        with tf.contrib.framework.arg_scope(
                [layers.convolution2d],
                kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,
                normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(self.weight_decay)
        ):
            self.X = tf.reshape(self.X, [-1, 28, 28, 1])

            net = layers.convolution2d(self.X, num_outputs=8, kernel_size=5, scope='conv1')
            net = layers.max_pool2d(net, kernel_size=2, scope='pool1')

            net = layers.convolution2d(net, num_outputs=16, kernel_size=5, scope='conv2')

            net = layers.flatten(net, [-1, 7 * 7 * 16])
            net = layers.fully_connected(net, num_outputs=32, activation_fn=tf.nn.tanh, scope='fc1')

            net = layers.fully_connected(net, num_outputs=self.class_num, scope='fc2')
            self.y = layers.softmax(net, scope='softmax')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.AdamOptimizer(step).minimize(self.loss)

        pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(pred, tf.float32))

        self.sess = tf.Session()

    def train(self, X_train, y_train, X_test, y_test, epochs=50000, batch_size=50):
        print("Starting to train")
        self.sess.run(tf.initialize_all_variables())

        batch_start = 0
        batch_end = batch_start + batch_size
        for iteration in range(epochs):
            _, loss, probs = self.sess.run(
                [self.optimizer, self.loss, self.y],
                feed_dict={self.X: X_train[batch_start:batch_end], self.y_: y_train[batch_start:batch_end]}
            )

            if iteration % 100 == 0:
                train_acc = self.sess.run(
                    self.acc,
                    feed_dict={self.X: X_train[batch_start:batch_end], self.y_: y_train[batch_start:batch_end]}
                )

                val_acc = self.sess.run(
                    self.acc,
                    feed_dict={self.X: X_test, self.y_: y_test}
                )

                print(
                    'Epoch: {}, train loss: {:2.4}, train acc: {:.2%}, validation acc: {:.2%}'.format(
                        iteration, loss, train_acc, val_acc)
                )

                if val_acc >= 0.995:
                    print('Validation acc is great')
                    break

            batch_start = batch_end
            batch_end += batch_size

            if batch_end > len(X_train):
                batch_start = 0
                batch_end = batch_start + batch_size
                X_train, y_train = self.__shuffle(X_train, y_train)

        print("Training ended")

    def __shuffle(self, a, b):
        p = np.random.permutation(len(a))
        return a[p], b[p]


if __name__ == '__main__':
    main()
