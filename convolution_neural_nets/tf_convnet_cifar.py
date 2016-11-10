import os
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers


def main():
    train_x, train_y, valid_x, valid_y, test_x, test_y = create_dataset('dataset', 32, 32, 3)

    print('Data shape:', train_x[0].shape)
    print('Train size:', len(train_x), 'Validation size:', len(valid_x), 'Test size:', len(test_x))

    feature_number = 32 * 32 * 3
    class_num = 10

    conv_net = TFConvNet(feature_number, class_num, False)
    conv_net.train(train_x, train_y, test_x, test_y)


def shuffle_data(data_x, data_y):
    indices = np.arange(data_x.shape[0])
    np.random.shuffle(indices)
    shuffled_data_x = np.ascontiguousarray(data_x[indices])
    shuffled_data_y = np.ascontiguousarray(data_y[indices])
    return shuffled_data_x, shuffled_data_y


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def to_one_hot(class_index, class_num=10):
    one_hot = [0.] * class_num
    one_hot[class_index] = 1.0
    return one_hot


def create_dataset(data_dir, img_height, img_width, num_channels):
    train_x = np.ndarray((0, img_height * img_width * num_channels), dtype=np.float32)
    train_y = []
    for i in range(1, 6):
        subset = unpickle(os.path.join(data_dir, 'data_batch_%d' % i))
        train_x = np.vstack((train_x, subset['data']))
        train_y += subset['labels']
    train_x = train_x.reshape((-1, img_height, img_width, num_channels))
    train_y = np.array(train_y, dtype=np.int32)

    subset = unpickle(os.path.join(data_dir, 'test_batch'))
    test_x = subset['data'].reshape((-1, img_height, img_width, num_channels)).astype(np.float32)
    test_y = np.array(subset['labels'], dtype=np.int32)

    valid_size = 5000
    train_x, train_y = shuffle_data(train_x, train_y)
    valid_x = train_x[:valid_size, ...]
    valid_y = train_y[:valid_size, ...]
    train_x = train_x[valid_size:, ...]
    train_y = train_y[valid_size:, ...]
    data_mean = train_x.mean((0, 1, 2))
    data_std = train_x.std((0, 1, 2))

    train_x = (train_x - data_mean) / data_std
    valid_x = (valid_x - data_mean) / data_std
    test_x = (test_x - data_mean) / data_std

    train_x = train_x.reshape([-1, 32, 32, 3])
    valid_x = valid_x.reshape([-1, 32, 32, 3])
    test_x = test_x.reshape([-1, 32, 32, 3])

    train_y = list(map(to_one_hot, train_y))
    valid_y = list(map(to_one_hot, valid_y))
    test_y = list(map(to_one_hot, test_y))

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
                kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.tanh,
                normalizer_fn=layers.batch_norm, normalizer_params=self.bn_params,
                weights_initializer=layers.variance_scaling_initializer(),
                weights_regularizer=layers.l2_regularizer(self.weight_decay)
        ):
            self.X = tf.reshape(self.X, [-1, 32, 32, 3])

            net = layers.convolution2d(self.X, num_outputs=8, kernel_size=[1, 5], scope='conv1')
            net = layers.convolution2d(net, num_outputs=8, kernel_size=[5, 1], scope='conv2')
            net = layers.max_pool2d(net, kernel_size=2, scope='pool1')

            net = layers.convolution2d(net, num_outputs=16, kernel_size=[1, 5], scope='conv4')
            net = layers.convolution2d(net, num_outputs=16, kernel_size=[5, 1], scope='conv5')
            net = layers.max_pool2d(net, kernel_size=2, scope='pool2')

            net = layers.flatten(net, [-1, 8 * 8 * 16])
            net = layers.fully_connected(net, num_outputs=32, activation_fn=tf.nn.tanh, scope='fc1')
            net = layers.fully_connected(net, num_outputs=self.class_num, scope='fc2')
            self.y = layers.softmax(net, scope='softmax')

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.y, self.y_))
        self.optimizer = tf.train.AdamOptimizer(step).minimize(self.loss)

        pred = tf.equal(tf.argmax(self.y, 1), tf.argmax(self.y_, 1))
        self.acc = tf.reduce_mean(tf.cast(pred, tf.float32))

        self.sess = tf.Session()

    def train(self, X_train, y_train, X_test, y_test, epochs=50000, batch_size=50):
        print("\nStarting to train")
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

        print("Training ended")


if __name__ == '__main__':
    main()
