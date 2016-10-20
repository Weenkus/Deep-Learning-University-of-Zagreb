import data
import numpy as np
import os
import matplotlib.pyplot as plt


def main():
    np.random.seed(100)

    X, Y_ = data.sample_gauss_2d(2, 100)
    w, b = binlogreg_train(X, Y_)

    probabilities = binlogreg_classify(X, w,b)
    Y = np.where(probabilities >= .5, 1, 0)

    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probabilities.argsort()])
    print('Acc: {0}\nRecall: {1}\nPrecision: {2}\nAP: {3}\n'.format(accuracy, recall, precision, AP))


def binlogreg_train(X, Y_, param_niter=500, param_delta=0.2, animate=False):
    b = 0
    w = np.random.randn(2)
    N = len(Y_)

    files = []
    for i in range(param_niter):
        scores = np.dot(X, w) + b   # result classification N x 1
        probabilities = 1. / (1 + np.exp(-scores))  # class probabilities c_1 # N x 1
        loss = np.sum(-np.log(probabilities))  # scalar

        if i % 20 == 0:
            print("iteration {}: loss {}".format(i, loss))

            if animate:
                # Graph
                Y = np.where(probabilities >= .5, 1, 0)
                bbox = (np.min(X, axis=0), np.max(X, axis=0))
                data.graph_surface(data.binlogreg_decfun(X, w, b), bbox, offset=0)
                data.graph_data(X, Y_, Y)

                # Animate
                file_name = '_tmp%03d.png' % i
                print('Saving frame', file_name)
                plt.savefig(file_name)
                files.append(file_name)

        dL_dscores = probabilities - Y_  # loss derivation on result classification N x 1

        grad_w = 1./N * np.dot(dL_dscores, X)   # D x 1
        grad_b = 1./N * np.sum(dL_dscores)   # 1 x 1

        w += -param_delta * grad_w
        b += -param_delta * grad_b

    if animate:
        print('Making movie animation.mpg - this make take a while')
        os.system("mencoder 'mf://_tmp*.png' -mf type=png:fps=10 -ovc lavc -lavcopts vcodec=wmv2 -oac copy -o animation.mpg")
        os.system("convert _tmp*.png animation.mng")

    # cleanup
    for file_name in files:
        os.remove(file_name)

    print('Weights:', w)
    return w, b


def binlogreg_classify(X, w, b):
    scores = np.dot(w, X.T) + b
    probabilities = 1. / (1 + np.exp(-scores))
    return probabilities

if __name__ == '__main__':
    main()
