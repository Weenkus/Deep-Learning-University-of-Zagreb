from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
import data


def main():
    np.random.seed(100)

    # Init the dataset
    class_num = 3
    X, Y_, Yoh_ = data.sample_gmm_2d(K=6, C=class_num, N=40)

    # Train the model
    svm = KSVMWrap(C=1.0, X=X, Y_=Y_, kernel='rbf')

    # Plot the results
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(svm_classify(X, svm), bbox, offset=0)
    data.graph_data(X, Y_, svm.predict(X), svm.get_scores())

    # show the results
    #plt.savefig('svm.png')
    plt.show()

    accuracy, recall, precision = data.eval_perf_binary(svm.predict(X), Y_)
    AP = 0.5
    print('Acc: {0}\nRecall: {1}\nPrecision: {2}\nAP: {3}\n'.format(accuracy, recall, precision, AP))

    svm.get_scores()


def svm_classify(X, model):
    def classify(X):
        return model.predict(X)
    return classify


class KSVMWrap(object):

    def __init__(self, C, X, Y_, kernel):
        self.X = X
        self.Y_ = Y_

        self.svm = SVC(C=C, kernel=kernel)
        self.svm.fit(X, Y_)

    def predict(self, X):
        return self.svm.predict(X)

    def get_scores(self):
        indecies = []
        for support_vector in self.svm.support_vectors_:
            for i, x in enumerate(self.X):
                if np.array_equal(x, support_vector):
                    indecies.append(i)

        return indecies


if __name__ == '__main__':
    main()
