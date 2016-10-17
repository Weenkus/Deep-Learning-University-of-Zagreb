import sympy
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import average_precision_score


def main():
  G = Random2DGaussian()
  X = G.get_sample(100)
  plt.scatter(X[:,0], X[:,1])
  plt.show()


class Random2DGaussian(object):
    np.random.seed(100)

    def __init__(self):
        self.min_x = 0
        self.max_x = 10
        self.min_y = 0
        self.max_y = 10

        centar_x = np.random.random_sample()
        centar_y = np.random.random_sample()
        self.mean = np.array([centar_x, centar_y])

        eigval_x = (np.random.random_sample()*(self.max_x - self.min_x)/5)**2
        eigval_y = (np.random.random_sample()*(self.max_y - self.min_y)/5)**2

        D = np.array([[eigval_x, 0], [0, eigval_y]])
        R = np.array([[45, 0], [0, 45]])

        self.covariance_matrix = R.T * D * R

    def get_sample(self, n, show=False):
        assert(n > 0)

        if show:
            print('Mean:\n', self.mean)
            print('\nCovariance matrix:\n', self.covariance_matrix)

        x, y = np.random.multivariate_normal(self.mean, self.covariance_matrix, size=n).T
        return np.column_stack((x, y))


def sample_gauss_2d(C, N):
    G = Random2DGaussian()
    Y_ = np.random.choice([0, 1], size=(N,), p=[1./2, 1./2])
    return G.get_sample(N), Y_


def eval_perf_binary(Y, Y_):
    return accuracy_score(Y_, Y), recall_score(Y_, Y), precision_score(Y_, Y)


def eval_AP(Y_sorted):
    return 0.5

if __name__ == '__main__':
    main()
