import numpy as np
from dataset import Parser


class RNN(object):

    def __init__(self, vocab_size, hidden_size=100, learning_rate=1e-1, init_factor=0.01,
                 gradient_clip_size=5, optimizer='AdaGrad', decay_rate=0.9):

        self.decay_rate = decay_rate
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
        x_one_hot = {}
        probs = {}
        h = {}
        h[-1] = np.copy(self.h)

        loss = self.__forward(x, y, x_one_hot, h, probs)
        dU, dW, dV, db, dc, dh_next = self.__backward(x, y, probs, h, x_one_hot)
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
        for param, gradient, gradient_memory in zip(
                [self.W, self.U, self.V, self.b, self.c],
                [dW, dU, dV, db, dc],
                [self.memory_W, self.memory_U, self.memory_V, self.memory_b, self.memory_c]
        ):

            gradient_memory += (self.decay_rate * gradient_memory) + ((1 - self.decay_rate) * (gradient * gradient))
            param += -self.learning_rate * gradient / np.sqrt(gradient_memory + 1e-8)

    def __adam(self, dU, dW, dV, db, dc):
        raise NotImplementedError

    def __forward(self, x, y, x_one_hot, h, probs):
        loss = 0
        y_true = y
        for time_step in range(len(x)):
            h_prev = h[time_step-1]
            x_one_hot[time_step] = np.zeros((self.vocab_size, 1))
            x_one_hot[time_step][x[time_step]] = 1

            h[time_step] = self.__hidden_output(x_one_hot[time_step], h_prev)
            out = self.__output(h[time_step])
            probs[time_step] = self.__softmax(out)

            loss += self.__loss(probs[time_step], y_true[time_step])

        return loss

    def __backward(self, x, y, probs, h, x_one_hot):
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)
        dh_next = np.zeros_like(self.h)

        for time_step in reversed(range(len(x))):
            y_grad = np.copy(probs[time_step])
            y_grad[y[time_step]] -= 1

            dV += np.dot(y_grad, h[time_step].T)
            dc += y_grad

            dh = np.dot(self.V.T, y_grad) + dh_next
            upstream_gradient = (1 - h[time_step]**2) * dh

            dU += np.dot(upstream_gradient, x_one_hot[time_step].T)
            dW += np.dot(upstream_gradient, h[time_step-1].T)
            db += upstream_gradient

            dh_next = np.dot(self.W.T, upstream_gradient)

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

    def sample(self, parser, seed, n_sample):
        char_ids = []
        h = self.h

        for char in seed:
            char_id = parser.encode(char)
            x_one_hot = np.zeros((self.vocab_size, 1))
            x_one_hot[char_id] = 1

            h = self.__hidden_output(x_one_hot, h)
            y = self.__output(h)
            probs = self.__softmax(y)
            char_id = np.random.choice(range(self.vocab_size), p=probs.ravel())

            x_one_hot = np.zeros((self.vocab_size, 1))
            x_one_hot[char_id] = 1
            char_ids.append(char_id)

        for t in range(n_sample):
            h = self.__hidden_output(x_one_hot, h)
            y = self.__output(h)
            probs = self.__softmax(y)
            char_id = np.random.choice(range(self.vocab_size), p=probs.ravel())

            x_one_hot = np.zeros((self.vocab_size, 1))
            x_one_hot[char_id] = 1
            char_ids.append(char_id)

        return char_ids


def language_model(vocab_size, parser, hidden_size, learning_rate, init_factor, epochs, gradient_clip_size,
                   optimizer, decay_rate, sequence_length):

    rnn = RNN(vocab_size, hidden_size, learning_rate, init_factor, gradient_clip_size, optimizer, decay_rate)

    losses = []
    for epoch in range(epochs):
        for i, x, y in parser.sequences_generator(sequence_length=sequence_length):

            if i % 1000 == 0:
                #seed = "HAN:\nIs that good or bad?\n\n"
                seed = 'H'
                char_ids = rnn.sample(parser, seed=seed, n_sample=200)
                txt = ''.join([parser.decode(n) for n in char_ids])
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
        gradient_clip_size=4,
        optimizer='AdaGrad',
        decay_rate=0.2,
        sequence_length=30
    )
