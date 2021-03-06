import numpy as np
from collections import Counter


class Parser(object):

    def __init__(self, file_path):
        assert file_path.endswith('.txt'), 'Parser only reads .txt files!'

        self.file_path = file_path
        self.encoding = "utf-8"
        self.batches = []

        self.batch_size = None
        self.sorted_chars = None
        self.char2id = None
        self.id2char = None
        self.x = None
        self.num_batches = None
        self.current_batch = 0

    def preprocess(self):
        with open(self.file_path, "r") as input_file:
            data = input_file.read().decode(self.encoding)
            counter_data = Counter(data)

        # count and sort most frequent characters
        sorted_chars = ''.join(letter * freq for letter, freq in counter_data.most_common())
        self.sorted_chars = sorted(set(sorted_chars), key=lambda x: sorted_chars.index(x))

        # self.sorted chars contains just the characters ordered descending by frequency
        self.char2id = dict(zip(self.sorted_chars, range(len(self.sorted_chars))))
        # reverse the mapping
        self.id2char = {k: v for v, k in self.char2id.items()}
        # convert the data to ids
        self.x = np.array(list(map(self.char2id.get, data)))

        print 'Processed input of size: {0}'.format(len(self.x))

    def encode(self, sequence):
        # returns the sequence encoded as integers
        return map(self.char2id.get, sequence)

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return self.id2char[encoded_sequence]

    def create_minibatches(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.num_batches = int(len(self.x) / (self.batch_size * sequence_length))

        print 'Number of batches: {0}'.format(self.num_batches)
        print 'Batch size: {0}'.format(self.batch_size)
        print 'Input sample: {0}...'.format(self.x[:20])

        for batch_start_index in range(0, self.num_batches, self.batch_size * sequence_length):
            batch_end_index = batch_start_index + (self.batch_size * sequence_length)
            X = np.array(self.x[batch_start_index:batch_end_index])
            y = np.array(self.x[batch_start_index+1:batch_end_index+1])

            X = map(lambda x: x if x is not None else 0, X)
            y = map(lambda x: x if x is not None else 0, y)

            data_shape = (batch_size, sequence_length)
            self.batches.append((np.reshape(X, data_shape), np.reshape(y, data_shape)))

    def minibatch_generator(self):
        for epoch, batch in enumerate(self.batches):
            batch_x, batch_y = batch
            yield epoch, batch_x, batch_y

    def sequences_generator(self, sequence_length):
        for i in range(len(self.x) / sequence_length):
            x = self.x[i * sequence_length:(i + 1) * sequence_length]
            y = self.x[i * sequence_length + 1:(i + 1) * sequence_length + 1]

            yield i, x, y

    def get_batch(self):
        batch_x, batch_y = self.batches[self.current_batch]
        self.current_batch = (self.current_batch + 1) % len(self.batches)

        return batch_x, batch_y

    def get_sequence(self):
        batch_x, batch_y = self.batches[self.current_batch]
        self.current_batch = (self.current_batch + 1) % len(self.batches)

        return batch_x[0], batch_y[0]

    def get_vocabulary_size(self):
        return len(self.sorted_chars)


def main():
    parser = Parser('data/selected_conversations.txt')
    parser.preprocess()
    parser.create_minibatches(batch_size=4, sequence_length=3)

    for i, batch in enumerate(parser.minibatch_generator()):
        print batch

        if i==5:
            exit(0)

if __name__ == '__main__':
    main()
