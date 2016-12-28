import numpy as np
import dataset
import time


class RNN(object):

    def __init__(self, hidden_size, sequence_length, vocab_size, learning_rate, batch_size):
        self.hidden_size = hidden_size
        self.sequence_length = sequence_length
        self.vocab_size = vocab_size
        self.learning_rate = learning_rate

        self.U = np.array(np.random.randn(vocab_size, hidden_size))
        self.W = np.array(np.random.randn(hidden_size, hidden_size))
        self.b = np.ones((hidden_size, batch_size))

        self.V = np.array(np.random.randn(hidden_size, vocab_size))
        self.c = np.ones((vocab_size, batch_size))

        # memory of past gradients - rolling sum of squares for Adagrad
        self.memory_U = np.zeros_like(self.U)
        self.memory_W = np.zeros_like(self.W)
        self.memory_V = np.zeros_like(self.V)

        self.memory_b, self.memory_c = np.zeros_like(self.b), np.zeros_like(self.c)

        self.grad_clip = 5

        print 'RNN created'
        for param_name, param in {'U': self.U, 'W': self.W, 'b': self.b, 'V': self.V, 'c': self.c}.iteritems():
            print param_name, ' ->', param.shape

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

        cache = (x, h_prev)

        h_temp = np.tanh(np.dot(x, U) + np.dot(h_prev, W))
        h_current = np.add(h_temp, b.T)


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
        for time_step in range(self.sequence_length):
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

        x, h_prev = cache

        dW = np.dot(grad_next.T, h_prev)
        dU = np.dot(grad_next.T, x).T
        db = grad_next.T
        dh_prev = np.dot(grad_next, self.W)

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
        for cache_step in reversed(cache):
            grad_next = dh * (1 - cache_step[1]**2)
            dh_prev_step, dU_step, dW_step, db_step = self.__rnn_step_backward(grad_next, cache_step)

            dU += dU_step
            dW += dW_step
            db += db_step

        # compute and return gradients with respect to each parameter
        # for the whole time series.
        # Why are we not computing the gradient with respect to inputs (x)?
        dU = np.clip(dU, -self.grad_clip, self.grad_clip)
        dW = np.clip(dW, -self.grad_clip, self.grad_clip)
        db = np.clip(db, -self.grad_clip, self.grad_clip)

        return dU, dW, db

    @staticmethod
    def __output(h, V, c):
        # Calculate the output probabilities of the network
        return np.add(np.dot(h, V), c.T)

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
        dh = 0
        dV = 0
        dc = 0
        for time_step, h_current in enumerate(h):
            output = self.__output(h_current, V, c)
            y_pred = self.softmax(output)

            y_true = y[:, time_step, :]

            y_grad = y_pred - y_true
            loss -= np.sum(y_true * np.log(y_pred))

            dc += y_grad.T
            dV += np.dot(h_current.T, y_grad)

            # TODO FIX dh
            dh += np.dot(V, y_pred.T).T
            #dh += y_true

        return loss, dh, dV, dc

    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return (e_x + 1) / (e_x.sum(axis=0) + 2)

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

        # update memory matrices
        # perform the Adagrad update of parameters
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        self.V -= self.learning_rate * dV
        self.c -= self.learning_rate * dc

    def sample(self, parser, seed, n_sample):
        h0 = np.zeros((1, self.hidden_size))

        seed_as_id = np.array([parser.encode(seed)]).reshape((1, len(seed)))
        seed_oh = np.array(map(self.one_hot, seed_as_id))

        h, cache = self.__rnn_forward(seed_oh, h0, self.U, self.W, self.b)

        sample = []
        for h_current in h:
            logits = self.__output(h_current, self.V, self.c)
            out = self.softmax(logits)
            out = out[0, :]
            char = parser.decode(np.argmax(out))
            sample.append(char)

        return ''.join(sample)


def run_language_model(max_epochs, hidden_size=100, sequence_length=30, learning_rate=1e-1, sample_every=100,
                       batch_size=16):
    parser = dataset.Parser('data/selected_conversations.txt')
    parser.preprocess()
    parser.create_minibatches(batch_size=batch_size, sequence_length=sequence_length)

    vocab_size = len(parser.sorted_chars)
    rnn = RNN(
        hidden_size=hidden_size,
        sequence_length=sequence_length,
        vocab_size=vocab_size,
        learning_rate=learning_rate,
        batch_size=batch_size)

    current_epoch = 0
    batch = 0

    h0 = np.zeros((batch_size, hidden_size))

    while current_epoch < max_epochs:
        losses = []
        for e, x, y in parser.minibatch_generator():
            if e == 0:
                current_epoch += 1
                h0 = np.zeros((batch_size, hidden_size))
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
                sample = rnn.sample(parser, seed="HAN:\nIs that good or bad?\n\n   ", n_sample=300)
                print sample
            batch += 1

        print 'Epoch: {0}, loss: {1}'.format(current_epoch, np.average(losses))


def main():
    run_language_model(max_epochs=30, learning_rate=1e-1, hidden_size=200, sequence_length=30, batch_size=1)

if __name__ == '__main__':
    main()
