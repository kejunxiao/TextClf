"""
data preprocessing and get batch
"""
import os
import re
import logging
import itertools
from collections import Counter
import numpy as np
import pandas as pd


class DataLoad(object):
    logging.getLogger().setLevel(logging.INFO)

    def __init__(self, data_path, batch_size, num_epochs, dev_sample_rate, forced_seq_len=None):
        """
        params:
            data_path: source data path
            mode: 'tarin' or 'dev'
            dev_sample_rate: percentage of the training data to use for validation
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.forced_seq_len = forced_seq_len
        self.dev_sample_rate = dev_sample_rate
        self._load_data()

    def train_batch_iter(self, shuffle=True):
        """
        params:

        returns:
        """
        x, y, data_size = self._split_train_dev('train')
        num_batchs_per_epoch = data_size // self.batch_size + 1

        for _ in range(self.num_epochs):
            if shuffle:
                shuffled_indices = np.random.permutation(np.arange(data_size))
                x, y = x[shuffled_indices], y[shuffled_indices]
            for i in range(num_batchs_per_epoch):
                start_idx = i * self.batch_size
                end_idx = min((i+1) * self.batch_size, data_size)
                yield x[start_idx:end_idx], y[start_idx:end_idx]

    def get_dev_data(self, shuffle=True):
        """
        params:

        returns:
        """
        dev_x, dev_y, dev_size = self._split_train_dev('dev')
        if shuffle:
            shuffled_indices = np.random.permutation(np.arange(dev_size))
            dev_x, dev_y = dev_x[shuffled_indices], dev_y[shuffled_indices]
        return dev_x, dev_y

    @staticmethod
    def _clean_str(s):
        s = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", s)
        s = re.sub(r" : ", ":", s)
        s = re.sub(r"\'s", " \'s", s)
        s = re.sub(r"\'ve", " \'ve", s)
        s = re.sub(r"n\'t", " n\'t", s)
        s = re.sub(r"\'re", " \'re", s)
        s = re.sub(r"\'d", " \'d", s)
        s = re.sub(r"\'ll", " \'ll", s)
        s = re.sub(r",", " , ", s)
        s = re.sub(r"!", " ! ", s)
        s = re.sub(r"\(", " \( ", s)
        s = re.sub(r"\)", " \) ", s)
        s = re.sub(r"\?", " \? ", s)
        s = re.sub(r"\s{2,}", " ", s)
        return s.strip().lower()

    def _load_data(self):
        """
        params:

        returns:
            x: 2D np.array
               samples, dimension is (N, self.forced_seq_len)
            y: 2D np.array
               labels, dimension is (N, len(labels))
            token2id: python dict object
            id2token: python dict object
            df: pd.DataFrame
            labels: 1D np.array
        """
        df = pd.read_csv(self.data_path)
        selected_cols = ['Descript', 'Category']
        df = df.loc[:, selected_cols].dropna(axis=0, how='any')

        # construct label one-hot vectors
        labels = np.unique(
            np.array(df.loc[:, selected_cols[1]], dtype=np.object))
        one_hot = np.zeros([len(labels), len(labels)], np.float)
        np.fill_diagonal(one_hot, 1)
        # {laebl: one hot vector for this label}
        labels2vec = dict(zip(labels, one_hot))

        raw_x = np.array(df.loc[:, selected_cols[0]].apply(
            lambda x: DataLoad._clean_str(x).split(' ')), dtype=np.object)
        raw_y = df.loc[:, selected_cols[1]].apply(
            lambda y: labels2vec[y]).tolist()

        # padding sentence
        padded_x = self._pad_sentence(raw_x)
        token2id = self._build_vocab(padded_x)
        x = []
        for sent in padded_x:
            xs = []
            for token in sent:
                if token not in token2id:
                    token = '<OOV>'
                xs.append(token2id[token])
            x.append(xs)
        self.x = np.array(x, dtype=np.int64)
        self.y = np.array(raw_y, dtype=np.float)

    def _split_train_dev(self, mode):
        # split data into train set or dev set
        data_size = self.x.shape[0]
        dev_size = int(data_size * self.dev_sample_rate)
        train_size = data_size - dev_size
        # maybe using cross-validation is better
        if mode == 'train':
            return self.x[:train_size], self.y[:train_size], train_size
        elif mode == 'dev':
            return self.x[dev_size:], self.y[dev_size:], dev_size
        else:
            raise ValueError('mode shoudle be train or dev.')

    def _pad_sentence(self, sentences, padding_word='<PAD>'):
        if self.forced_seq_len is None:
            # forced_seq_len = max length of all sentences
            self.forced_seq_len = max([len(sent) for sent in sentences])
        padded_sentences = []
        for sent in sentences:
            if len(sent) < self.forced_seq_len:
                sent.extend([padding_word] * (self.forced_seq_len-len(sent)))
                padded_sent = sent
            elif len(sent) > self.forced_seq_len:
                logging.info('Because the length of the sentence is larger the self.forced_seq_len,'
                             'so need to cut off the sentence.')
                padded_sent = sent[:self.forced_seq_len]
            padded_sentences.append(padded_sent)
        return padded_sentences

    def _build_vocab(self, sentences):
        tokens_count = Counter(itertools.chain(*sentences))
        vocab = [token[0]
                 for token in tokens_count.most_common(self.forced_seq_len)]
        vocab += ['<OOV>']  # out of vocablary
        token2id = {token: i for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        return token2id


if __name__ == '__main__':
    params = {
        'data_path': '../dataset/San_Francisco_Crime/train.csv.zip',
        'batch_size': 32,
        'num_epochs': 200,
        'forced_seq_len': 14,
        'dev_sample_rate':0.05
    }
    data = DataLoad(data_path=params['data_path'],
                    batch_size=params['batch_size'],
                    num_epochs=params['num_epochs'],
                    forced_seq_len=params['forced_seq_len'],
                    dev_sample_rate=params['dev_sample_rate'])

    batches = data.train_batch_iter()
    batch_x, batch_y = next(batches)
    # print(len(batches))
    print(batch_x.shape)
    print(batch_y.shape)
