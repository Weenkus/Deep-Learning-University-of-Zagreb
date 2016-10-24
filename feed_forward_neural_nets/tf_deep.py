import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import data


def main():
    np.random.seed(100)
    tf.set_random_seed(100)

    # Init the dataset
    class_num = 3
    X, Y_, Yoh_ = data.sample_gmm_2d(K=6, C=class_num, N=40)

    # Construct the computing graph
    tf_deep = TFDeep(
        nn_configuration=[2, 10, 10, class_num],
        param_delta=0.1,
        param_lambda=1e-4,
        no_linearity_function=tf.nn.tanh
    )

    tf_deep.count_params()

    tf_deep.train(X, Yoh_, param_niter=1000)
    tf_deep.eval(X, Yoh_)

    # Plot the results
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(td_classify(X, tf_deep), bbox, offset=0)
    data.graph_data(X, Y_, tf_deep.predict(X))

    # show the results
    #plt.savefig('tf_deep_data_sigmoid.png')
    plt.show()


def td_classify(X, model):
    def classify(X):
        return model.predict(X)
    return classify


class TFDeep(object):

    def __init__(self, nn_configuration, param_delta, param_lambda, no_linearity_function, adam=False, decay=False):
        self.nn_configuration = nn_configuration
        self.param_lambda = param_lambda
        self.dimension_num = nn_configuration[0]
        self.class_num = nn_configuration[-1]
        self.activation_function = no_linearity_function

        self.X = tf.placeholder(tf.float32, [None, self.dimension_num])
        self.Yoh_ = tf.placeholder(tf.float32, [None, self.class_num])

        self.__construct_layers(nn_configuration)
        self.__construct_output()

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(self.pred, self.Yoh_))
        for layer_component in self.W + self.b:
            self.loss += param_lambda * tf.nn.l2_loss(layer_component)

        if decay:
            param_delta = tf.train.exponential_decay(
                learning_rate=param_delta,
                global_step=100 * 100,
                decay_steps=56000,
                decay_rate=0.95,
                staircase=True
            )

        if adam:
            self.optimizer = tf.train.AdamOptimizer(param_delta).minimize(self.loss)
        else:
            self.optimizer = tf.train.GradientDescentOptimizer(param_delta).minimize(self.loss)

        self.sess = tf.Session()

    def __construct_layers(self, nn_configuration):
        self.W = []
        self.b = []
        print('Layer structure:')

        for i, layer_dimension in enumerate(nn_configuration[:-1]):
            layer_input = layer_dimension
            layer_output = nn_configuration[i+1]

            print('Layer W{0}: [{1}, {2}]  B{0}: [{2}]'.format(i, layer_input, layer_output))
            self.W.append(tf.Variable(tf.random_normal([layer_input, layer_output]), name='W{0}'.format(i)))
            self.b.append(tf.Variable(tf.random_normal([layer_output]), name='b{0}'.format(i)))

        print()

    def __construct_output(self):
        current_layer_input = self.X
        for layer_index, weight in enumerate(self.W):
            regularization = self.param_lambda * tf.nn.l2_loss(weight)
            layer = tf.add(tf.matmul(current_layer_input, weight), self.b[layer_index]) + regularization

            layer = self.activation_function(layer) if layer_index != len(self.W) - 1 else tf.nn.softmax(layer)
            current_layer_input = layer

        self.pred = current_layer_input

    def count_params(self):
        print('Params:')
        for var in tf.trainable_variables():
            print(var.name, tf.shape(var))

        param_per_layer = self.nn_configuration[0] + 1
        total_param = sum([param_per_layer * layer for layer in self.nn_configuration[1:]])
        print('Total parameter components:', total_param)
        print()

    def train(self, X, Yoh_, param_niter):
        init = tf.initialize_all_variables()
        self.sess.run(init)

        for iteration in range(param_niter):

            _, loss, pred, weights = self.sess.run(
                [self.optimizer, self.loss, self.pred, self.W],
                feed_dict={self.X: X, self.Yoh_: Yoh_}
            )

            self.weights = weights
            print('Iteration: {0}, loss: {1}'.format(iteration, loss))

    def train_mb(self, mnist, epoch_number, batch_size):
        init = tf.initialize_all_variables()
        self.sess.run(init)

        for iteration in range(epoch_number):
            batch_num = int(mnist.train.num_examples/batch_size)
            avg_loss = 0

            for i in range(batch_num):
                batch_x, batch_y = mnist.train.next_batch(batch_size)

                _, loss, pred, weights = self.sess.run(
                    [self.optimizer, self.loss, self.pred, self.W],
                    feed_dict={self.X: batch_x, self.Yoh_: batch_y}
                )

                avg_loss += loss / batch_num

            self.weights = weights
            print('Epoch: {0}, loss: {1}'.format(iteration, avg_loss))


    def eval(self, X, Yoh_):
        correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.Yoh_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({self.X: X, self.Yoh_: Yoh_}, session=self.sess))
        return self.pred

    def predict(self, X):
        probs = self.sess.run(self.pred, feed_dict={self.X: X})
        return np.argmax(probs, axis=1)

    def get_weights(self):
        return self.weights


if __name__ == '__main__':
    main()
