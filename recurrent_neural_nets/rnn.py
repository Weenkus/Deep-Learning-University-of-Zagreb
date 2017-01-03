import numpy as np
from dataset import Parser


class RNN(object):

    def __init__(self, vocab_size, hidden_size=100, learning_rate=1e-1, init_factor=0.01,
                 gradient_clip_size=5, optimizer='AdaGrad', decay_rate=0.9):

        self.optimizer = optimizer
        self.gradient_clip_size = gradient_clip_size
        self.vocab_size = vocab_size
        self.h = np.zeros((hidden_size, 1))

        self.W = init_factor * np.random.randn(hidden_size, hidden_size)
        self.U = init_factor * np.random.randn(hidden_size, vocab_size)
        self.V = init_factor * np.random.randn(vocab_size, hidden_size)
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((vocab_size, 1))

        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)
        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

        self.learning_rate = learning_rate

        self.optimizers = {
            'GradientDescent': self.__gradient_descent,
            'AdaGrad': self.__adagrad,
            'RMSProp': self.__rmsprop,
            'Adam': self.__adam
        }

    def train(self, x, y):
        xhat = {}
        probs = {}
        h = {}
        h[-1] = np.copy(self.h)

        loss = self.__forward(x, y, xhat, h, probs)
        dU, dW, dV, db, dc, dh_next = self.__backward(x, y, probs, h, xhat)
        self.__update_params(dU, dW, dV, db, dc)

        self.h = h[len(x)-1]

        return loss

    def __update_params(self, dU, dW, dV, db, dc):
        self.__gradient_clip([dU, dW, dV, db, dc])
        self.optimizers[self.optimizer](dU, dW, dV, db, dc)

    def __gradient_descent(self, dU, dW, dV, db, dc):
        for param, gradient in zip([self.W, self.U, self.V, self.b, self.c], [dW, dU, dV, db, dc]):
            param -= self.learning_rate * gradient

    def __adagrad(self, dU, dW, dV, db, dc):
        for param, gradient, gradient_memory in zip(
                [self.W, self.U, self.V, self.b, self.c],
                [dW, dU, dV, db, dc],
                [self.memory_W, self.memory_U, self.memory_V, self.memory_b, self.memory_c]
        ):

            gradient_memory += gradient * gradient
            param += -self.learning_rate * gradient / np.sqrt(gradient_memory + 1e-8)

    def __rmsprop(self, dU, dW, dV, db, dc):
        raise NotImplementedError

    def __adam(self, dU, dW, dV, db, dc):
        raise NotImplementedError

    def __forward(self, x, y, xhat, h, probs):
        loss = 0
        y_true = y
        for time_step in range(len(x)):
            h_prev = h[time_step-1]
            xhat[time_step] = np.zeros((self.vocab_size, 1))
            xhat[time_step][x[time_step]] = 1

            h[time_step] = self.__hidden_output(xhat[time_step], h_prev)
            out = self.__output(h[time_step])
            probs[time_step] = self.__softmax(out)

            loss += self.__loss(probs[time_step], y_true[time_step])

        return loss

    def __backward(self, x, y, probs, h, xhat):
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)
        dh_next = np.zeros_like(self.h)

        for time_step in reversed(range(len(x))):
            #backprop into y. see http://cs231n.github.io/neural-networks-case-study/#grad if confused here
            dy = np.copy(probs[time_step])
            dy[y[time_step]] -= 1

            #find updates for y
            dV += np.dot(dy, h[time_step].T)
            dc += dy

            #backprop into h and through tanh nonlinearity
            dh = np.dot(self.V.T, dy) + dh_next
            dh_raw = (1 - h[time_step]**2) * dh

            #find updates for h
            dU += np.dot(dh_raw, xhat[time_step].T)
            dW += np.dot(dh_raw, h[time_step-1].T)
            db += dh_raw

            #save dh_next for subsequent iteration
            dh_next = np.dot(self.W.T, dh_raw)

        return dU, dW, dV, db, dc, dh_next

    def __gradient_clip(self, gradients):
        for gradient in gradients:
            np.clip(gradient, -self.gradient_clip_size, self.gradient_clip_size, out=gradient)

    def __softmax(self, y_pred):
        return np.exp(y_pred) / np.sum(np.exp(y_pred))

    def __output(self, h):
        return np.dot(self.V, h) + self.c

    def __hidden_output(self, x, h):
        return np.tanh(np.dot(self.U, x) + np.dot(self.W, h) + self.b)

    def __loss(self, prob, y_true):
        return -np.log(prob[y_true, 0])

    def sample(self, seed, n):
        ndxs = []
        h = self.h

        xhat = np.zeros((self.vocab_size, 1))
        xhat[seed] = 1#transform to 1-of-k

        for t in range(n):
            h = np.tanh(np.dot(self.U, xhat) + np.dot(self.W, h) + self.b)#update the state
            y = np.dot(self.V, h) + self.c
            p = np.exp(y) / np.sum(np.exp(y))
            ndx = np.random.choice(range(self.vocab_size), p=p.ravel())

            xhat = np.zeros((self.vocab_size, 1))
            xhat[ndx] = 1

            ndxs.append(ndx)

        return ndxs


def language_model(vocab_size, parser, hidden_size, learning_rate, init_factor, epochs, gradient_clip_size,
                   optimizer, decay_rate, sequence_length):

    rnn = RNN(vocab_size, hidden_size, learning_rate, init_factor, gradient_clip_size, optimizer, decay_rate)

    losses = []
    for epoch in range(epochs):
        for i, x, y in parser.sequences_generator(sequence_length=sequence_length):

            if i % 1000 == 0:
                sample_ix = rnn.sample(x[0], 200)
                txt = ''.join([parser.decode(n) for n in sample_ix])
                print txt

            loss = rnn.train(x, y)

            if i % 1000 == 0:
                print '(%d)iteration %d, loss = %f' % (epoch+1, i, loss)
                losses.append(loss)


if __name__ == "__main__":
    parser = Parser('data/selected_conversations.txt')
    parser.preprocess()

    language_model(
        vocab_size=parser.get_vocabulary_size(),
        parser=parser,
        hidden_size=100,
        learning_rate=1e-1,
        init_factor=2e-2,
        epochs=16,
        gradient_clip_size=5,
        optimizer='AdaGrad',
        decay_rate=0.9,
        sequence_length=30
    )
