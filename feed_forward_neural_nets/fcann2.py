import matplotlib.pyplot as plt
import numpy as np
import data


def main():
    # create the dataset
    X, Y_ = data.sample_gmm_2d(K=6, C=2, N=10)
    model = fcann2_train(X, Y_)

    # fit the model
    probabilities = fcann2_classify(X, model)
    Y = np.argmax(probabilities, axis=1)

    # evaluate the model
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    AP = data.eval_AP(Y_[probabilities.argsort()])
    print('Acc: {0}\nRecall: {1}\nPrecision: {2}\nAP: {3}\n'.format(accuracy, recall, precision, AP))

    # graph the data points
    bbox = (np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(fcann2_decfun(X, model), bbox, offset=0)
    data.graph_data(X, Y_, Y)

    # show the results
    #plt.savefig('fcann2_classification.png')
    plt.show()


def fcann2_train(X, Y_, param_niter=10000, param_delta=0.00005, param_lambda=1e-3, hidden_layer_dim=15):
    np.random.seed(100)

    output_dim = 2
    input_dim = 2
    N = len(Y_)

    b1 = np.ones((1, hidden_layer_dim))
    W1 = np.array(np.random.randn(input_dim, hidden_layer_dim))

    b2 = np.ones((1, output_dim))
    W2 = np.array(np.random.randn(hidden_layer_dim, output_dim))

    neural_net = {
        'W1': W1,
        'b1': b1,
        'W2': W2,
        'b2': b2
    }

    for i in range(param_niter):
        # Forward propagation
        probabilities = fcann2_classify(X, neural_net)

        # Back propagation
        delta3 = probabilities
        delta3[range(N), Y_] -= 1

        scores1 = X.dot(neural_net['W1']) + neural_net['b1']
        #h1 = np.tanh(scores1)
        h1 = np.maximum(0, scores1)

        dW2 = np.transpose(h1).dot(delta3)
        db2 = np.sum(delta3, axis=0, keepdims=True)

        #delta2 = delta3.dot(W2.T) * (1 - np.power(h1, 2))
        delta2 = np.dot(delta3, W2.T)
        delta2[h1 <= 0] = 0

        dW1 = np.dot(X.T, delta2)
        db1 = np.sum(delta2, axis=0, keepdims=True)

        # Regularization
        dW2 += param_lambda * W2
        dW1 += param_lambda * W1

        # Optimize params
        W1 += -param_delta * dW1
        b1 += -param_delta * db1
        W2 += -param_delta * dW2
        b2 += -param_delta * db2

        neural_net = {
            'W1': W1,
            'b1': b1,
            'W2': W2,
            'b2': b2
        }

        if i % 10 == 0:
            corect_logprobs = -np.log(probabilities[range(N), Y_])
            data_loss = np.sum(corect_logprobs)
            loss = 1./N * data_loss
            print("iteration {}: loss {}".format(i, loss))

    print('\n\n                   *****  Neural Net   *****\n')
    print('W1: {0}\nb1: {1}\nW2: {2}\nb2: {3}\n'.format(
        neural_net['W1'], neural_net['b1'], neural_net['W2'], neural_net['b1']
    ))
    return neural_net


def fcann2_classify(X, model):
    # Forward propagation
    scores1 = X.dot(model['W1']) + model['b1']
    #h1 = np.tanh(scores1)
    h1 = np.maximum(0, scores1)
    scores2 = h1.dot(model['W2']) + model['b2']

    # Output
    exp_scores = np.exp(scores2)
    probabilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    return probabilities


def fcann2_decfun(X, model):
    def classify(X):
        probabilities = fcann2_classify(X, model)
        Y = np.argmax(probabilities, axis=1)
        return probabilities[0][Y]
    return classify


if __name__ == '__main__':
    main()
