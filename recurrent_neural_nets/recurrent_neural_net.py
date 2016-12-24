import numpy as np


class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.array(np.random.randn(vocab_size, hidden_size))
        self.W = np.array(np.random.randn(hidden_size, hidden_size))
        self.b = np.ones((1, hidden_size))

        self.V = np.array(np.random.randn(hidden_size, vocab_size))
        self.c = np.ones((1, vocab_size))

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)

        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

        print 'RNN created'
        for param_name, param in {'U': self.U, 'W': self.W, 'b': self.b, 'V': self.V, 'c': self.c}.iteritems():
            print param_name, ' ->', param.shape


def main():
    rnn = RNN(hidden_size=100, sequence_length=30, vocab_size=70, learning_rate=1e-1)


if __name__ == '__main__':
    main()
