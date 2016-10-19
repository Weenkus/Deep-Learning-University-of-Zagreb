import data
import numpy as np


def main():
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(2, 100)
    w, b = binlogreg_train(X, Y_)

    probabilities = binlogreg_classify(X, w,b)
    Y = np.where(probabilities >= .5, 1, 0)

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probabilities.argsort()])
    print('Acc: {0}\nRecall: {1}\nPrecision: {2}\nAP: {3}\n'.format(accuracy, recall, precision, AP))


def binlogreg_train(X, Y_, param_niter=1000, param_delta=0.2):
    b = 0
    w = np.random.randn(2)
    N = len(Y_)

    for i in range(param_niter):
        scores = np.dot(X, w) + b   # klasifikacijski rezultati N x 1
        probabilities = 1. / (1 + np.exp(-scores))  # vjerojatnosti razreda c_1 # N x 1
        loss = np.sum(-np.log(probabilities))  # scalar

        if i % 10 == 0:
            print("iteration {}: loss {}".format(i, loss))

        dL_dscores = probabilities - Y_  # derivacije gubitka po klasifikacijskom rezultatu N x 1

        grad_w = 1./N * np.dot(dL_dscores, X)   # D x 1
        grad_b = 1./N * np.sum(dL_dscores)   # 1 x 1

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    print('Weights:', w)
    return w, b


def binlogreg_classify(X, w, b):
    scores = np.dot(w, X.T) + b
    probabilities = 1. / (1 + np.exp(-scores))
    return probabilities

if __name__ == '__main__':
    main()
