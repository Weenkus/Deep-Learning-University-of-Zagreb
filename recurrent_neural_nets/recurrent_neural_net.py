import numpy as np
import dataset


class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate, decay_rate=None,
                 optimizier=None, init_factor=0.01, grad_clip=5):

        self.optimizer = optimizier
        self.optimizers = {
            'Adam': self.__adam,
            'AdaGrad': self.__adagrad,
            'GradientDescent': self.__gradient_descent,
            'RMSProp': self.__rmsprop
        }

        self.decay_rate = decay_rate
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = init_factor * np.random.randn(vocab_size, hidden_size)
        self.W = init_factor * np.random.randn(hidden_size, hidden_size)
        self.b = np.zeros((hidden_size, 1))

        self.V = init_factor * np.random.randn(hidden_size, vocab_size)
        self.c = np.zeros((vocab_size, 1))

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)

        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

        self.grad_clip = grad_clip

        print 'RNN created'
        for param_name, param in {'U': self.U, 'W': self.W, 'b': self.b, 'V': self.V, 'c': self.c}.iteritems():
            print param_name, ' ->', param.shape

    def __gradient_descent(self, dU, dW, dV, db, dc):
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        self.V -= self.learning_rate * dV
        self.c -= self.learning_rate * dc

    def __adagrad(self, dU, dW, dV, db, dc):
        self.memory_U += np.multiply(dU, dU)
        self.memory_W += np.multiply(dW, dW)
        self.memory_V += np.multiply(dV, dV)
        self.memory_b += np.multiply(db, db)
        self.memory_c += np.multiply(dc, dc)

        update_U = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_U)), dU)
        update_W = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_W)), dW)
        update_V = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_V)), dV)
        update_b = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_b)), db)
        update_c = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_c)), dc)

        self.U += update_U
        self.W += update_W
        self.V += update_V
        self.b += update_b
        self.c += update_c

    def __rmsprop(self, dU, dW, dV, db, dc):
        self.memory_U = (self.decay_rate * self.memory_U) + ((1 - self.decay_rate) * np.multiply(dU, dU))
        self.memory_W = (self.decay_rate * self.memory_W) + ((1 - self.decay_rate) * np.multiply(dW, dW))
        self.memory_V = (self.decay_rate * self.memory_V) + ((1 - self.decay_rate) * np.multiply(dV, dV))
        self.memory_b = (self.decay_rate * self.memory_b) + ((1 - self.decay_rate) * np.multiply(db, db))
        self.memory_c = (self.decay_rate * self.memory_c) + ((1 - self.decay_rate) * np.multiply(dc, dc))

        update_U = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_U)), dU)
        update_W = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_W)), dW)
        update_V = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_V)), dV)
        update_b = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_b)), db)
        update_c = np.multiply(-self.learning_rate / (1e-7 + np.sqrt(self.memory_c)), dc)

        self.U += update_U
        self.W += update_W
        self.V += update_V
        self.b += update_b
        self.c += update_c

    def __adam(self):
        raise NotImplementedError

    def step(self, h0, x_oh, y_oh):
        h, cache = self.__rnn_forward(x_oh, h0, self.U, self.W, self.b)
        loss, dh, dV, dc = self.__output_loss_and_grads(h, self.V, self.c, y_oh)
        dU, dW, db = self.__rnn_backward(dh, cache)
        self.__update(dU, dW, db, dV, dc)

        return loss, h[-1]

    def __rnn_step_forward(self, x, h_prev, U, W, b):
        # A single time step forward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # x - input data (minibatch size x input dimension)
        # h_prev - previous hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)
        h_current = np.tanh(np.dot(x, U) + np.dot(h_prev.T, W) + b.T).T

        cache = (x, h_current, h_prev)

        # return the new hidden state and a tuple of values needed for the backward step
        return h_current, cache

    def __rnn_forward(self, x, h0, U, W, b):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        # x - input data for the whole time-series (minibatch size x sequence_length x input dimension)
        # h0 - initial hidden state (minibatch size x hidden size)
        # U - input projection matrix (input dimension x hidden size)
        # W - hidden to hidden projection matrix (hidden size x hidden size)
        # b - bias of shape (hidden size x 1)

        cache = []
        h = []
        h_previous = h0
        seq_len = x.shape[1]
        for time_step in range(seq_len):
            h_current, cache_current = self.__rnn_step_forward(x[:, time_step], h_previous, U, W, b)
            h_previous = h_current

            cache.append(cache_current)
            h.append(h_current)

        # return the hidden states for the whole time series (T+1) and a tuple of values needed for the backward step
        return h, cache

    def __rnn_step_backward(self, grad_next, cache):
        # A single time step backward of a recurrent neural network with a
        # hyperbolic tangent nonlinearity.

        # grad_next - upstream gradient of the loss with respect to the next hidden state and current output
        # cache - cached information from the forward pass

        x, h_current, h_prev = cache
        dW = np.dot(grad_next, h_prev.T)
        dU = np.dot(grad_next, x).T
        db = grad_next
        dh_prev = np.dot(grad_next.T, self.W)

        # compute and return gradients with respect to each parameter
        # HINT: you can use the chain rule to compute the derivative of the
        # hyperbolic tangent function and use it to compute the gradient
        # with respect to the remaining parameters

        return dh_prev, dU, dW, db

    def __rnn_backward(self, dh, cache):
        # Full unroll forward of the recurrent neural network with a
        # hyperbolic tangent nonlinearity

        dU = 0
        dW = 0
        db = 0
        for cache_step, dh_step in zip(reversed(cache), dh):
            x, h_current, h_prev = cache_step

            grad_next = dh_step.T * (1 - h_current**2)
            dh_prev_step, dU_step, dW_step, db_step = self.__rnn_step_backward(grad_next, cache_step)

            dU_step = np.clip(dU_step, -self.grad_clip, self.grad_clip)
            dW_step = np.clip(dW_step, -self.grad_clip, self.grad_clip)
            db_step = np.clip(db_step, -self.grad_clip, self.grad_clip)

            dU += dU_step
            dW += dW_step
            db += db_step

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?

        return dU, dW, db

    @staticmethod
    def __output(h, V, c):
        # Calculate the output probabilities of the network
        return np.dot(h.T, V) + c.T

    def __output_loss_and_grads(self, h, V, c, y):
        # Calculate the loss of the network for each of the outputs

        # h - hidden states of the network for each timestep.
        #     the dimensionality of h is
        #     (batch size x sequence length x hidden size (the initial state is irrelevant for the output)
        #
        # V - the output projection matrix of dimension hidden size x vocabulary size
        # c - the output bias of dimension vocabulary size x 1
        #
        # y - the true class distribution - a one-hot vector of dimension
        #     vocabulary size x 1 - you need to do this conversion prior to
        #     passing the argument. A fast way to create a one-hot vector from
        #     an id could be something like the following code:

        # y[timestep] = np.zeros((vocabulary_size, 1))
        # y[timestep][batch_y[timestep]] = 1

        #     where y might be a dictionary.

        # calculate the output (o) - unnormalized log probabilities of classes
        # calculate yhat - softmax of the output
        # calculate the cross-entropy loss
        # calculate the derivative of the cross-entropy softmax loss with respect to the output (o)
        # calculate the gradients with respect to the output parameters V and c
        # calculate the gradients with respect to the hidden layer h

        loss = 0
        dV = 0
        dc = 0
        dh_previous = 0
        dh = []
        for time_step, h_current in enumerate(reversed(h)):
            y_pred = self.softmax(self.__output(h_current, V, c))

            current_step = len(h) - time_step - 1
            y_true = y[:, current_step, :]
            y_grad = y_pred - y_true

            N = h_current.shape[0]

            prob = y_pred[0][np.where(y_true[0] == 1)][0]
            loss -= np.average(np.log(prob))

            dc += y_grad.T
            dV += np.dot(h_current, y_grad)

            dh_current = np.dot(V, y_grad.T) + dh_previous
            dh_previous = dh_current

            # dV = np.divide(dV, self.batch_size)
            # dc = np.divide(dc, self.batch_size)
            # dh_current = np.divide(dh_current, self.batch_size)

            dh_current = np.clip(dh_current, -self.grad_clip, self.grad_clip)
            dh.append(dh_current.T)

        dc = np.clip(dc, -self.grad_clip, self.grad_clip)
        dV = np.clip(dV, -self.grad_clip, self.grad_clip)

        # dh is a list of hidden state gradients in reverse order dh[0] is the gradient of the last time step
        return loss, dh, dV, dc

    @staticmethod
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        y_pred = exp_x / np.sum(exp_x, axis=1, keepdims=True)

        # Check the softmax gives probabilities
        for time_step in range(x.shape[0]):
            assert abs(1 - sum(y_pred[time_step, :])) < 1e-6

        return y_pred

    def one_hot(self, vector):
        """
        FROM: http://stackoverflow.com/questions/29831489/numpy-1-hot-array

        Converts an input 1-D vector of integers into an output
        2-D array of one-hot vectors, where an i'th input value
        of j will set a '1' in the i'th row, j'th column of the
        output array.

        Example:
            v = np.array((1, 0, 4))
            one_hot_v = convertToOneHot(v)
            print one_hot_v

            [[0 1 0 0 0]
             [1 0 0 0 0]
             [0 0 0 0 1]]
        """

        assert isinstance(vector, np.ndarray)
        assert len(vector) > 0
        num_classes = self.vocab_size

        if num_classes is None:
            num_classes = np.max(vector) + 1
        else:
            assert num_classes > 0
            assert num_classes >= np.max(vector)

        result = np.zeros(shape=(len(vector), num_classes))
        result[np.arange(len(vector)), vector] = 1
        return result.astype(int)

    def __update(self, dU, dW, db, dV, dc):
        db = np.array([np.average(db, axis=1)]).T
        dc = np.array([np.average(dc, axis=1)]).T

        self.optimizers[self.optimizer](dU, dW, dV, db, dc)

    def sample(self, parser, seed, n_sample):
        h0 = np.zeros((self.hidden_size, 1))

        seed_as_id = np.array([parser.encode(seed)]).reshape((1, len(seed)))
        seed_as_id = np.array([map(lambda x: x if x is not None else 0, seed_as_id[0])])
        input = np.array(map(self.one_hot, seed_as_id))
        h, cache = self.__rnn_forward(input, h0, self.U, self.W, self.b)
        h_prev = h[-1]

        out = self.softmax(self.__output(h_prev, self.V, self.c))
        char_id = np.random.choice(range(self.vocab_size), p=out.ravel())
        char = parser.decode(char_id)

        output = []
        for i in range(n_sample):
            char_as_id = np.array([parser.encode(char)])
            char_one_hot = self.one_hot(char_as_id)
            h_current, cache = self.__rnn_step_forward(char_one_hot, h_prev, self.U, self.W, self.b)

            out = self.softmax(self.__output(h_current, self.V, self.c))
            char_id = np.random.choice(range(self.vocab_size), p=out.ravel())
            char = parser.decode(char_id)
            output.append(char)

            h_prev = h_current

        return ''.join(output)


def run_language_model(max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100,
                       batch_size=16, decay_rate=0.9, optimizer='GradientDescent', init_factor=0.01, grad_clip=5):
    parser = dataset.Parser('data/selected_conversations.txt')
    parser.preprocess()
    parser.create_minibatches(batch_size=batch_size, sequence_length=sequence_length)

    vocab_size = len(parser.sorted_chars)
    rnn = RNN(
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        decay_rate=decay_rate,
        optimizier=optimizer,
        init_factor=init_factor,
        grad_clip=grad_clip
    )

    current_epoch = 0
    batch = 0

    h0 = np.zeros((hidden_size, batch_size))

    while current_epoch < max_epochs:
        losses = []
        for e, x, y in parser.minibatch_generator():
            if e == 0:
                current_epoch += 1
                h0 = np.zeros((hidden_size, batch_size))
                # why do we reset the hidden state here?

            # One-hot transform the x and y batches
            x_oh = np.array(map(rnn.one_hot, x))
            y_oh = np.array(map(rnn.one_hot, y))

            # Run the recurrent network on the current batch
            # Since we are using windows of a short length of characters,
            # the step function should return the hidden state at the end
            # of the unroll. You should then use that hidden state as the
            # input for the next minibatch. In this way, we artificially
            # preserve context between batches.

            loss, h0 = rnn.step(h0, x_oh, y_oh)
            losses.append(loss)

            if batch % sample_every == 0:
                sample = rnn.sample(parser, seed="HAN:\nIs that good or bad?\n\n", n_sample=100)
                print 'RNN:', sample
            batch += 1

        print 'Epoch: {0}, loss: {1}'.format(current_epoch, np.average(losses))


def main():
    run_language_model(
        max_epochs=10000,
        learning_rate=1e-1,
        hidden_size=100,
        sequence_length=30,
        batch_size=128,
        sample_every=300,
        decay_rate=0.95,
        optimizer='AdaGrad',
        init_factor=0.02,
        grad_clip=5
    )

if __name__ == '__main__':
    main()
