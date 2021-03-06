import numpy as np
import math
import matplotlib.pyplot as plt
import random
import binlogreg

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score


def main():
    np.random.seed(100)

    # create the dataset
    X, Y_ = sample_gauss_2d(2, 100)
    w, b = binlogreg.binlogreg_train(X, Y_, animate=False)

    # fit the model
    probabilities = binlogreg.binlogreg_classify(X, w, b)
    Y = np.where(probabilities >= .5, 1, 0)

    # evaluate the model
    accuracy, recall, precision = eval_perf_binary(Y, Y_)
    AP = eval_AP(Y_[probabilities.argsort()])
    print('Acc: {0}\nRecall: {1}\nPrecision: {2}\nAP: {3}\n'.format(accuracy, recall, precision, AP))

    # graph the data points
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    graph_surface(binlogreg_decfun(X, w, b), bbox, offset=0)
    graph_data(X, Y_, Y)

    # show the results
    #plt.savefig('Binary_logistic_regression_classification.png')
    #plt.show()


class Random2DGaussian(object):

    def __init__(self):
        self.min_x = 0
        self.max_x = 10
        self.min_y = 0
        self.max_y = 10

        centar_x = (self.max_x - self.min_x) * np.random.random_sample() + self.min_x
        centar_y = (self.max_y - self.min_y) * np.random.random_sample() + self.min_y
        self.mean = np.array([centar_x, centar_y])

        eigval_x = (np.random.random_sample() * (self.max_x - self.min_x)/5)**2
        eigval_y = (np.random.random_sample() * (self.max_y - self.min_y)/5)**2
        D = np.array([[eigval_x, 0], [0, eigval_y]])

        theta = random.randint(-180, 180)
        R = np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])

        self.cov = np.dot(np.dot(R.T, D), R)

    def get_sample(self, n, show=False):
        assert(n > 0)

        if show:
            print('Mean:', self.mean)
            print('Covariance matrix:\n', self.cov)
            print()

        x, y = np.random.multivariate_normal(self.mean, self.cov, n).T
        return np.column_stack((x, y))


def sample_gauss_2d(C, N):
    class_size = math.ceil(N / C)

    X_parts = []
    for i in range(0, C):
        normally_distributed_data = Random2DGaussian().get_sample(class_size, show=True)
        X_parts.append(normally_distributed_data)

    X = np.vstack((X_parts[0], X_parts[1]))

    Y_ = np.full((class_size, 1), 0)
    for i in range(1, C):
        i_class = np.full((class_size, 1), i)
        Y_ = np.vstack((Y_, i_class))

    return X, Y_.ravel()


def eval_perf_binary(Y, Y_):
    return accuracy_score(Y_, Y), recall_score(Y_, Y), precision_score(Y_, Y)


def eval_AP(Y_sorted):
    return 0.5


def graph_data(X, Y_, Y):
    predictions = ['o' if y == y_ else 's' for y, y_ in zip(Y_, Y)]
    for i, prediction in enumerate(predictions):

        color = 'grey' if Y_[i] == 0 else 'white'
        plt.scatter(X[i][0], X[i][1], marker=prediction, s=40, c=color)


def myDummyDecision(X):
    scores = X[:, 0] + X[:, 1] - 5
    return scores > 0.5


def binlogreg_decfun(X, w, b):
    def classify(X):
        return binlogreg.binlogreg_classify(X, w, b)
    return classify


def graph_surface(fun, rect, offset=0.5, width=1000, height=800, resolution=400):
    x_min, x_max = rect[0][0], rect[1][0]
    y_min, y_max = rect[0][1], rect[1][1]

    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    coordinate_net = np.vstack((xx.ravel(), yy.ravel())).T

    Z = fun(coordinate_net)
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(16, 9))
    plt.pcolormesh(xx, yy, Z, vmin=0, vmax=1)
    plt.colorbar()

if __name__ == '__main__':
    main()
