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
        outputs = self.embed() # outputs dimension is (N, hp.FORCED_SEQ_LEN, hp.EMBED_SIZE, 1)
        outputs = self.conv2d_banks(outputs)
        pass


    def embed(self, scope='embedding', reuse=None):
        """
        word embedding.
        """
        with tf.device('/cpu:0'), tf.variable_scope(scope=scope, reuse=reuse):
            lookup_table = tf.get_variable(name='lookup_table', 
                                        dtype=tf.float32, 
                                        shape=[hp.VOCAB_SIZE, hp.EMBED_SIZE], 
                                        initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.01))
            embeded_inputs = tf.nn.embedding_lookup(lookup_table, self.inputs)
        return tf.expand_dims(embeded_inputs, -1) # add channel

    def conv2d(self, 
               inputs, 
               filter_size, 
               num_filters, 
               max_pool_size, 
               scope='conv2d-maxpooling', 
               reuse=None):
        with tf.variable_scope(scope=scope+'-{}'.format(filter_size), reuse=reuse):
            filter_shape = [filter_size, hp.EMBED_SIZE, 1, num_filters] #HWIO
            W = tf.get_variable(name='W-conv2d-%s' % filter_size, 
                                shape=filter_shape, 
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            b = tf.get_variable(name='b-conv2d-%s' % filter_size,
                                shape=[num_filters],
                                dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))
            conv = tf.nn.conv2d(input=inputs,
                                filter=W,
                                strides=[1,1,1,1],
                                padding='VALID',
                                name='conv')
            h = tf.nn.relu(tf.nn.bias_add(conv, b), name='relu')
            
            # maxpooling 
            pooled = tf.nn.max_pool(value=h, 
                                    ksize=[1, max_pool_size, 1, 1], 
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='maxpool')
        return pooled

    def conv2d_banks(self, inputs, scope='banks', reuse=None):
        with tf.variable_scope(scope=scope, reuse=reuse):
            outputs = []
            reduced = np.int32(np.ceil(hp.FORCED_SEQ_LEN * 1.0 / hp.MAX_POOL_SIZE))
            for filter_size in hp.FILTERS_SIZE:
                pooled = self.conv2d(inputs=inputs,
                                     filter_size=filter_size,
                                     num_filters=hp.NUM_FILTERS,
                                     max_pool_size=hp.FORCED_SEQ_LEN - filter_size + 1)
                outputs.append(pooled)
            # residual connection
            outputs = tf.concat(outputs, axis=3)
            # dropout
            outputs = tf.nn.dropout(outputs, self.dropout_keep_prob)
        return outputs

    def output_layer(self):
        pass

            



# def batch_norm(inputs, 
#                data_format='NWC', 
#                is_training=True, 
#                activation_fn=None, 
#                scope='batch_norm', 
#                reuse=None):
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
#                                            activation_fn=activation_fn,
#                                            center=True,
#                                            scale=True,
#                                            is_training=is_training,
#                                            reuse=reuse,
#                                            fused=True,
#                                            data_format=data_format,
#                                            scope=scope)
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
        
# def conv1d(inputs, 
#            filters=None,
#            kernel_size=3,
#            stride=1,
#            padding='SAME',
#            activation_fn=None,
#            scope='conv1d', 
#            reuse=None):
#     with tf.variable_scope(scope=scope):
#         if filters is None:
#             filters = inputs.get_shape().as_list()[-1]
#         if activation_fn is None:
#             activation_fn = tf.nn.relu
#         outputs = tf.layers.conv1d(inputs=inputs,
#                                    filters=filters,
#                                    kernel_size=kernel_size,
#                                    strides=stride,
#                                    padding=padding,
#                                    activation=activation_fn,
#                                    reuse=reuse)
#     return outputs

# def conv1d_banks(inputs, K=16, is_training=True, scope='conv1d_banks', reuse=True):
#     with tf.variable_scope(scope=scope, reuse=reuse):
#         outputs = tf.layers.conv1d(inputs, filters=hp.EMBED_SIZE//2, kernel_size=3, strides=1, padding='SAME')
#         for k in range(2, K+1):
#             with tf.variable_scope(scope='con1d_banks_num{}'.format(k)):
#                 output = conv1d(inputs, hp.EMBED_SIZE//2)
#                 outputs = tf.concat([outputs, output], axis=-1)
#         outputs = batch_norm(outputs, is_training=is_training, activation_fn=tf.nn.relu)
#     return outputs

# def outputs_layer(inputs, scope='outputs_layer', reuse=None):
#     with tf.variable_scope(scope=scope, reuse=reuse):
#         outputs = tf.layers.dense(inputs=inputs,
#                                   units=hp.NUM_CLASSES,
#                                   activation=tf.nn.tanh)
#     return outputs
        
