"""
nerual nexwork
"""
import numpy as np
import tensorflow as tf
import hyperparams as hp

class Network(object):
    def __init__(self):
        self.inputs = tf.placeholder(dtype=tf.int32, shape=[None, hp.FORCED_SEQ_LEN],name='inputs')
        self.labels = tf.placeholder(dtype=tf.int32, shape=[None, hp.NUM_CLASSES], name='labels')
        self.dropout_keep_prob = tf.placeholder(dtype=tf.float32, name='dropout_keep_prob')

    def construct_network(self):
        self.embed() # outputs dimension is (N, hp.FORCED_SEQ_LEN, hp.EMBED_SIZE, 1)
        self.conv2d_banks()
        self.output_layer()

        # loss and optimizer
        with tf.variable_scope(scope='loss'):
            self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.labels))
            optimizer = tf.train.AdamOptimizer(learning_rate=hp.LEARNING_RATE)
            self.train_op = optimizer.minimize(self.loss_op)

        # accuracy
        with tf.variable_scope(scope='accuracy'):
            correct_preds = tf.equal(self.preds, tf.argmax(self.labels, axis=1))
            self.accuracy_op = tf.reduce_mean(tf.cast(correct_preds, tf.float32))

    def embed(self, scope='embedding', reuse=None):
        """
        word embedding.
        """
        with tf.device('/cpu:0'), tf.variable_scope(scope=scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table', 
                                        dtype=tf.float32, 
                                        shape=[hp.VOCAB_SIZE, hp.EMBED_SIZE], 
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            embedded_inputs = tf.nn.embedding_lookup(lookup_table, self.inputs)
            self.embedded_expanded_tokens = tf.expand_dims(embeded_inputs, -1) # add channel
        return self

    def conv2d_banks(self, scope='banks', reuse=None):
        with tf.variable_scope(scope=scope, reuse=reuse):
            outputs = []
            for filter_size in hp.FILTERS_SIZE_LIST:
                pooled = Network.conv2d(inputs=self.embedded_expanded_tokens,
                                        filter_size=filter_size,
                                        max_pool_size=hp.FORCED_SEQ_LEN - filter_size + 1,
                                        padding='VALID')
                # pooled dimension is (N, 1, 1, num_filters)
                outputs.append(pooled)
            # combine all pooled features
            h_pool = tf.concat(outputs, axis=3)
            num_filters_total = hp.NUM_FILTERS * len(hp.FILTERS_SIZE_LIST)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])
            # dropout
            self.droped_h = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)
        return self

    def output_layer(self, scope='output', reuse=None):
        tf.variable_scope(scope=scope, reuse=reuse):
            W = tf.get_variable(name='W',
                                shape=[inputs.get_shape().as_list()[-1], hp.NUM_CLASSES],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            b = tf.get_variable(name='b',
                                shape=[hp.NUM_CLASSES],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            self.logits = tf.nn.xw_plus_b(self.droped_h, W, b, name='logits')
            self.preds = tf.nn.softmax(self.scores, axis=1, name='preds')
        return self

    @staticmethod
    def conv2d(inputs,
               filter_size, 
               scope='conv2d-maxpooling',
               padding='VALID',
               reuse=None):
        with tf.variable_scope(scope=scope+'-{}'.format(filter_size), reuse=reuse):
            filter_shape = [filter_size, hp.EMBED_SIZE, 1, hp.NUM_FILTERS] #HWIO
            W = tf.get_variable(name='W', 
                                shape=filter_shape, 
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable(name='b',
                                shape=[hp.NUM_FILTERS],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            conv = tf.nn.conv2d(input=inputs,
                                filter=W,
                                strides=[1,1,1,1],
                                padding=padding,
                                name='conv')
            # nonlinear transition
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            
            # maxpooling 
            if padding == 'VALID':
                max_pool_size = hp.FORCED_SEQ_LEN - filter_size + 1
            elif padding == 'SAME':
                max_pool_size = hp.FORCED_SEQ_LEN + filter_size + 1
            pooled = tf.nn.max_pool(value=h, 
                                    ksize=[1, max_pool_size, 1, 1], 
                                    strides=[1, 1, 1, 1],
                                    padding=padding,
                                    name='maxpool')
        return pooled

    # def batch_norm(inputs, 
    #             data_format='NWC', 
    #             is_training=True, 
    #             activation_fn=None, 
    #             scope='batch_norm', 
    #             reuse=None):
    #     """
    #     batch normlize to accelerate convergence.
    #     params:
    #         inputs: a tensor with 3 rank and the data_format is NWC or NCW.
    #     return:
    #     """
    #     # using fused_batch_norm as it is much faster.
    #     # pay attention to the fact that fused_batch_norm requires shape to be rank 4 of NHWC.
    #     if data_format == 'NWC':
    #         inputs = tf.expand_dims(inputs, axis=1)
    #         data_format = 'NHWC'
    #     elif data_format == 'NCW':
    #         inputs = tf.expand_dims(inputs, axis=2)
    #         data_format = 'NCHW'
    #     else:
    #         raise ValueError('the data format must be NWC or NCW')
        
    #     outputs = tf.contrib.layers.batch_norm(inputs=inputs,
    #                                         activation_fn=activation_fn,
    #                                         center=True,
    #                                         scale=True,
    #                                         is_training=is_training,
    #                                         reuse=reuse,
    #                                         fused=True,
    #                                         data_format=data_format,
    #                                         scope=scope)
    #     return outputs

    # def prenet(inputs, num_units=None, is_training=True, scope='prenet', reuse=None):
    #     """
    #     a list non-linear transition for inputs
    #     params:
    #         inputs: a 3D tensor, shape=[batch_size, text_length, embed_size]
    #     return:
    #     """
    #     with tf.variable_scope(scope=scope, reuse=reuse):
    #         if num_units is None:
    #             num_units = [hp.EMBED_SIZE, hp.EMBED_SIZE//2]
    #         outputs = tf.layers.dense(inputs=inputs, units=num_units[0], activation=tf.nn.relu, name='dense1')
    #         outputs = tf.layers.dropout(inputs=outputs, rate=hp.DROPOUT_RATE, training=is_training, name='dropout1')
    #         outputs = tf.layers.dense(inputs=outputs, units=num_units[1], activation=tf.nn.relu, name='dense2')
    #         outputs = tf.layers.dropout(inputs=outputs, rate=hp.DROPOUT_RATE, training=is_training, name='dropout2')
    #     return outputs

    # def lstm(inputs, num_units=None, bidirection=False, scope='lstm', reuse=None):
    #     with tf.variable_scope(scope=scope, reuse=reuse):
    #         if num_units is None:
    #             num_units = inputs.get_shape().as_list()[-1]
    #         cell = tf.contrib.rnn.LSTMCell(num_units=num_units)
    #         if bidirection:
    #             cell_bw = tf.contrib.rnn.LSTMCell(num_units=num_units)
    #             outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell, cell_bw, inputs, dtype=tf.float32)
    #             return tf.concat(outputs, axis=1)
    #         else:
    #             outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    #             return outputs

if __name__ == '__main__':
    nw = Network()