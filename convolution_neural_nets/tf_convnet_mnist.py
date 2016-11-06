import time
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

from tensorflow.examples.tutorials.mnist import input_data


tf.app.flags.DEFINE_string('data_dir',  '/tmp/data/', 'Directory for storing data')


def main():
    dataset = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
    train_x, train_y, valid_x, valid_y, test_x, test_y = create_dataset(dataset)

    feature_number = dataset.train.images.shape[1]
    class_num = dataset.train.labels.shape[1]

    X = tf.placeholder(tf.float32, [None, feature_number])
    y_ = tf.placeholder(tf.float32, [None, class_num])

    nn = build_model(train_x, train_y, 10, False)

    sess = tf.Session()
    init = tf.initialize_all_variables()
    sess.run(init)

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(train_y, y_))
    optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    for iteration in range(10):
        _, loss = sess.run([optimizer, loss], feed_dict={X: train_x, y_: train_y})
        print('Iteration: {0}, loss: {1}'.format(iteration, loss))


def create_dataset(dataset):
    np.random.seed(int(time.time() * 1e6) % 2 ** 31)

    train_x = dataset.train.images
    train_x = train_x.reshape([-1, 1, 28, 28])
    train_y = dataset.train.labels

    valid_x = dataset.validation.images
    valid_x = valid_x.reshape([-1, 1, 28, 28])
    valid_y = dataset.validation.labels

    test_x = dataset.test.images
    test_x = test_x.reshape([-1, 1, 28, 28])
    test_y = dataset.test.labels

    train_mean = train_x.mean()
    train_x -= train_mean
    valid_x -= train_mean
    test_x -= train_mean

    return train_x, train_y, valid_x, valid_y, test_x, test_y


def build_model(inputs, labels, num_classes, is_training):
    weight_decay = 1e-3
    bn_params = {
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

    with tf.contrib.framework.arg_scope(
        [layers.convolution2d],
        kernel_size=3, stride=1, padding='SAME', activation_fn=tf.nn.relu,
        normalizer_fn=layers.batch_norm, normalizer_params=bn_params,
        weights_initializer=layers.variance_scaling_initializer(),
        weights_regularizer=layers.l2_regularizer(weight_decay)
    ):

        net = layers.convolution2d(inputs, num_outputs=16, kernel_size=5, scope='conv1')
        net = layers.max_pool2d(net, scope='pool1')
        net = layers.relu(net, scope='relu1')

        net = layers.convolution2d(net, num_outputs=32, kernel_size=5, scope='conv2')
        #net = layers.flatten(net, scope='flatten1')

        net = layers.fully_connected(net, scope='fc1')
        net = layers.relu(net, scope='relu2')

        net = layers.fully_connected(net, scope='fc2')
        net = layers.softmax(net, scope='softmax')

        return net

if __name__ == '__main__':
    main()
