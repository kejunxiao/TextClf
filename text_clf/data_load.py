"""
data preprocessing and get batch
"""
import os
import re
import logging
import itertools
import numpy as np
import pandas as pd
import hyperparams as hp
from collections import Counter


class DataLoad(object):
    logging.getLogger().setLevel(logging.INFO)

    def __init__(self, data_path):
        self.data_path =  data_path

    def load_data(self):
        """
        params:

        returns:
            x: 2D np.array
               samples, dimension is (N, hp.FORCED_SEQ_LEN)
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
        labels = np.unique(np.array(df.loc[:, selected_cols[1]], dtype=np.object))
        one_hot = np.zeros([len(labels), len(labels)], np.int32)
        np.fill_diagonal(one_hot, 1)
        # {laebl: one hot vector for this label}
        labels2vec = dict(zip(labels, one_hot))
        
        raw_x = np.array(df.loc[:, selected_cols[0]].apply(lambda x: DataLoad.clean_str(x).split(' ')))
        y = np.array(df.loc[:, selected_cols[1]].apply(lambda y: labels2vec[y]))

        # padding sentence
        paded_x = DataLoad.pad_sentence(raw_x)
        token2id, vocab = DataLoad.build_vocab(paded_x)
        x = []
        for sent in paded_x:
            xs = []
            for token in sent:
                if token not in token2id:
                    token = '<OOV>'
                xs.append([token2id[token]])
            x.append(np.array(xs))
        x = np.array(x)
        return x, y, token2id, vocab, df, labels

    def batch_iter(self, shuffle=True):
        x, y, _, _, _, _ = self.load_data()
        data_size = x.shape[0]
        num_batchs_per_epoch = data_size // hp.BATCH_SIZE + 1

        for _ in range(hp.NUM_EPOCHS):
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                x = x[shuffle_indices]
                y = y[shuffle_indices]
            for i in range(num_batchs_per_epoch):
                start_idx = i * hp.BATCH_SIZE
                end_idx = min((i+1)*hp.BATCH_SIZE, data_size)
                yield x[start_idx:end_idx], y[start_idx:end_idx]


    @staticmethod
    def clean_str(s):
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

    @staticmethod
    def pad_sentence(sentences, padding_word='<PAD>'):
        if not hp.FORCED_SEQ_LEN:
            seq_len = max([len(sent) for sent in sentences])
        else:
            seq_len = hp.FORCED_SEQ_LEN
        print(seq_len)
        paded_sentences = []
        for sent in sentences:
            if len(sent) < seq_len:
                paded_sent = sent + [padding_word]*(seq_len-len(sent))
            elif len(sent) > seq_len:
                logging.info('Because the length of the sentence is larger the hp.FORCED_SEQ_LEN,'
                             'so need to slice the sentence.')
                paded_sent = sent[:seq_len]
            paded_sentences.append(paded_sent)
        return paded_sentences

    @staticmethod
    def build_vocab(sentences):
        tokens_count = Counter(itertools.chain(*sentences))
        vocab = [token[0] for token in tokens_count.most_common(hp.FORCED_SEQ_LEN)]
        vocab += ['<OOV>'] # out of vocablary
        token2id = {token:i for i, token in enumerate(vocab)}
        return token2id, vocab


if __name__ == '__main__':
    dl = DataLoad(hp.DATA_PATH)
    dl.load_data()
    # for _ in range(3):
    #     print(next(dl.batch_iter()))