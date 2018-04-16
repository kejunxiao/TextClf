"""
nerual nexwork
"""
import tensorflow as tf


class TextCNN(object):
    def __init__(self, forced_seq_len, vocab_size, embedding_size, filters_size_list,
                 num_filters, num_classes, l2_reg_lambda=0.1):
        self.inputs = tf.placeholder(dtype=tf.int64, shape=[
                                     None, forced_seq_len], name='inputs')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[
                                     None, num_classes], name='labels')
        self.dropout_keep_prob = tf.placeholder(
            dtype=tf.float32, name='dropout_keep_prob')

        # model params
        self.forced_seq_len = forced_seq_len
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.filters_size_list = filters_size_list
        self.num_filters = num_filters
        self.num_classes = num_classes
        self.l2_reg_lambda = l2_reg_lambda

        self.construct_network()

    def construct_network(self):
        self.embed()  # outputs dimension is (N, self.forced_seq_len, self.embedding_size, 1)
        self.conv2d_banks()
        self.outputs()

        # loss
        with tf.variable_scope('loss'):
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.logits, labels=self.labels))
            self.loss_op += self.l2_reg_lambda * self.l2_loss

        # accuracy
        with tf.variable_scope('accuracy'):
            correct_preds = tf.equal(tf.argmax(self.preds, axis=1),
                                     tf.argmax(self.labels, axis=1))
            self.accuracy_op = tf.reduce_mean(
                tf.cast(correct_preds, tf.float32))

    def embed(self, scope='embedding', reuse=None):
        """
        word embedding.
        """
        with tf.device('/cpu:0'), tf.variable_scope(scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table',
                                           dtype=tf.float32,
                                           shape=[self.vocab_size,
                                                  self.embedding_size],
                                           initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            embedded_inputs = tf.nn.embedding_lookup(lookup_table, self.inputs)
            self.embedded_expanded_sents = tf.expand_dims(
                embedded_inputs, -1)  # add channel
        return self

    def conv2d(self,
               inputs,
               filter_size,
               scope='conv2d-maxpooling',
               padding='VALID',
               reuse=None):
        with tf.variable_scope(scope+'-{}'.format(filter_size), reuse=reuse):
            filter_shape = [filter_size, self.embedding_size,
                            1, self.num_filters]  # HWIO
            W = tf.get_variable(name='W',
                                shape=filter_shape,
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable(name='b',
                                shape=[self.num_filters],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            conv = tf.nn.conv2d(input=inputs,
                                filter=W,
                                strides=[1, 1, 1, 1],
                                padding=padding,
                                name='conv')
            # nonlinear transition
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')

            # maxpooling
            if padding == 'VALID':
                max_pool_size = self.forced_seq_len - filter_size + 1
            elif padding == 'SAME':
                max_pool_size = self.forced_seq_len + filter_size + 1
            pooled = tf.nn.max_pool(value=h,
                                    ksize=[1, max_pool_size, 1, 1],
                                    strides=[1, 1, 1, 1],
                                    padding=padding,
                                    name='maxpool')
        return pooled

    def conv2d_banks(self, scope='banks', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            outputs = []
            for filter_size in self.filters_size_list:
                pooled = self.conv2d(inputs=self.embedded_expanded_sents,
                                     filter_size=filter_size,
                                     padding='VALID')
                # pooled dimension is (N, 1, 1, num_filters)
                outputs.append(pooled)
            # combine all pooled features
            h_pool = tf.concat(outputs, axis=3)
            num_filters_total = self.num_filters * len(self.filters_size_list)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # dropout
            self.droped_h = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return self

    def outputs(self, scope='outputs', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[self.droped_h.get_shape().as_list()
                                       [-1], self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[self.num_classes],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.l2_loss = tf.nn.l2_loss(W)
            self.l2_loss += tf.nn.l2_loss(b)
            self.logits = tf.nn.xw_plus_b(self.droped_h, W, b, name='logits')
            self.preds = tf.nn.softmax(self.logits, axis=1, name='preds')
        return self
