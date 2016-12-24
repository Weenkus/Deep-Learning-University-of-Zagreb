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
        return map(self.char2id, sequence)

    def decode(self, encoded_sequence):
        # returns the sequence decoded as letters
        return map(self.id2char, encoded_sequence)

    def create_minibatches(self, batch_size, sequence_length):
        self.batch_size = batch_size
        self.num_batches = int(len(self.x) / (self.batch_size * sequence_length))

        print 'Number of batches: {0}'.format(self.num_batches)

        for batch_start_index in range(0, self.num_batches, self.batch_size):
            batch_end_index = batch_start_index + self.batch_size

            data_X = self.x[batch_start_index:batch_end_index]
            data_y = self.x[batch_start_index+1:batch_end_index+1]

            self.batches.append([data_X, data_y])

    def minibatch_generator(self):
        for epoch, batch in enumerate(self.batches):
            batch_x, batch_y = batch
            yield epoch, batch_x, batch_y


def main():
    parser = Parser('data/selected_conversations.txt')
    parser.preprocess()
    parser.create_minibatches(batch_size=16, sequence_length=1)

    for batch in parser.minibatch_generator():
        print batch

if __name__ == '__main__':
    main()
