import tensorflow as tf
import numpy as np
import tf_deep as td
import data
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data

tf.app.flags.DEFINE_string('data_dir',  '/tmp/data/', 'Directory for storing data')


def main():
    np.random.seed(100)
    tf.set_random_seed(100)

    mnist = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=True)
    minst_no_oh = input_data.read_data_sets(tf.app.flags.FLAGS.data_dir, one_hot=False)

    example_number = mnist.train.images.shape[0]
    feature_number = mnist.train.images.shape[1]
    class_num = mnist.train.labels.shape[1]

    X = mnist.train.images
    Yoh_ = mnist.train.labels
    Y_ = minst_no_oh.train.labels

    Xtest = mnist.test.images
    Ytest_ = minst_no_oh.test.labels

    print(example_number, feature_number, class_num)

    # Construct the computing graph
    deep_model = td.TFDeep(
        nn_configuration=[feature_number, 100, class_num],
        param_delta=0.5,
        param_lambda=1e-4,
        no_linearity_function=tf.nn.tanh
    )

    deep_model.train(X, Yoh_, param_niter=5000)
    deep_model.eval(X, Yoh_)

    accuracy, recall, precision = data.eval_perf_binary(deep_model.predict(X), Y_)
    print('TRAINING\nAcc: {0}\nRecall: {1}\nPrecision: {2}\n'.format(accuracy, recall, precision))

    accuracy, recall, precision = data.eval_perf_binary(deep_model.predict(Xtest), Ytest_)
    print('\nTEST\nAcc: {0}\nRecall: {1}\nPrecision: {2}\n'.format(accuracy, recall, precision))

    print_all_numbers(deep_model)


def print_all_numbers(model, class_num=10):
    weights = model.get_weights()
    numbers = [weights[0][:, i].reshape(28, 28) for i in range(class_num)]

    for i in range(class_num):
        plt.imshow(numbers[i], cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
        plt.savefig('numbers{0}.png'.format(i))

    plt.show()


if __name__ == '__main__':
    main()
