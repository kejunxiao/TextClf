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
import params


class DataLoad(object):
    logging.getLogger().setLevel(logging.INFO)

    def __init__(self, data_path, mode, batch_size, num_epochs, forced_seq_len=None,
                 dev_sample_rate=0.2):
        """
        params:
            data_path: source data path
            mode: 'tarin' or 'dev'
            dev_sample_rate: percentage of the training data to use for validation
        """
        self.data_path = data_path
        self.mode = mode
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.forced_seq_len = forced_seq_len
        self.dev_sample_rate = dev_sample_rate
        self._load_data()

    def batch_iter(self, shuffle=True):
        """
        params:

        returns:
        """
        x, y, data_size = self._split_train_dev()
        num_batchs_per_epoch = data_size // self.batch_size + 1

        for _ in range(self.num_epochs):
            if shuffle:
                shuffled_indices = np.random.permutation(np.arange(data_size))
                x, y = x[shuffled_indices], y[shuffled_indices]
            for i in range(num_batchs_per_epoch):
                start_idx = i * self.batch_size
                end_idx = min((i+1) * self.batch_size, data_size)
                yield x[start_idx:end_idx], y[start_idx:end_idx]

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
        one_hot = np.zeros([len(labels), len(labels)], np.int32)
        np.fill_diagonal(one_hot, 1)
        # {laebl: one hot vector for this label}
        labels2vec = dict(zip(labels, one_hot))

        raw_x = np.array(df.loc[:, selected_cols[0]].apply(
            lambda x: DataLoad._clean_str(x).split(' ')))
        self.y = np.array(df.loc[:, selected_cols[1]].apply(
            lambda y: labels2vec[y]))

        # padding sentence
        paded_x = self._pad_sentence(raw_x)
        token2id = self._build_vocab(paded_x)
        x = []
        for sent in paded_x:
            xs = []
            for token in sent:
                if token not in token2id:
                    token = '<OOV>'
                xs.append([token2id[token]])
            x.append(np.array(xs))
        self.x = np.array(x)

    def _split_train_dev(self):
        # split train set or dev set
        data_size = self.x.shape[0]
        dev_size = int(data_size * self.dev_sample_rate)
        train_size = data_size - dev_size
        # maybe using cross-validation is better
        if self.mode == 'train':
            return self.x[:train_size], self.y[:train_size], train_size
        elif self.mode == 'dev':
            return self.x[dev_size:], self.y[dev_size:], dev_size
        else:
            raise ValueError('mode shoudle be train or dev.')

    def _pad_sentence(self, sentences, padding_word='<PAD>'):
        if self.forced_seq_len is None:
            # forced_seq_len = max length of all sentences
            self.forced_seq_len = max([len(sent) for sent in sentences])
        paded_sentences = []
        for sent in sentences:
            if len(sent) < self.forced_seq_len:
                paded_sent = sent + [padding_word] * \
                    (self.forced_seq_len-len(sent))
            elif len(sent) > self.forced_seq_len:
                logging.info('Because the length of the sentence is larger the self.forced_seq_len,'
                             'so need to cut off the sentence.')
                paded_sent = sent[:self.forced_seq_len]
            paded_sentences.append(paded_sent)
        return paded_sentences

    def _build_vocab(self, sentences):
        tokens_count = Counter(itertools.chain(*sentences))
        vocab = [token[0]
                 for token in tokens_count.most_common(self.forced_seq_len)]
        vocab += ['<OOV>']  # out of vocablary
        token2id = {token: i for i, token in enumerate(vocab)}
        self.vocab_size = len(vocab)
        return token2id


if __name__ == '__main__':
    train_data = DataLoad(data_path=params.DATA_PATH,
                          mode='train',
                          batch_size=params.BATCH_SIZE,
                          num_epochs=params.NUM_EPOCHS,
                          forced_seq_len=params.FORCED_SEQ_LEN)
    print(next(train_data.batch_iter()))
