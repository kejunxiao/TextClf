import os
import time
import tensorflow as tf
from data_load import DataLoad
from model import TextCNN

# Parameters
# Data loading parameters
tf.app.flags.DEFINE_float("dev_sample_rate", .2,
                          "Percentage of the training data to use for validation")
tf.app.flags.DEFINE_string("train_data_path",
                           "../dataset/San_Francisco_Crime/train.csv.zip",
                           "Data source for the train data.")

# Model Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 5e-3,
                          "learning rate (default:0.001)")
tf.app.flags.DEFINE_integer("embedding_size", 128,
                            "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_list("filters_size_list", [3, 4, 5],
                         "list type filter sizes (default: [3, 4, 5])")
tf.app.flags.DEFINE_integer("num_filters", 8,
                            "Number of filters per filter size (default: 8)")
tf.app.flags.DEFINE_float("dropout_keep_prob", .5,
                          "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_integer("num_classes", 39,
                            "number of classes (default: 39)")

# Training schemes
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 200,
                            "Number of training epochs (default: 200)")
tf.app.flags.DEFINE_integer("evaluate_every", 100,
                            "Evaluate model on dev set after this many steps (default: 100)")

# Misc Parameters
tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                            "Allow device soft device placement")
tf.app.flags.DEFINE_boolean("log_device_placement", False,
                            "Log placement of ops on devices")

FLAGS = tf.app.flags.FLAGS

# training
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # loading training data
        data = DataLoad(data_path=FLAGS.train_data_path,
                        batch_size=FLAGS.batch_size,
                        num_epochs=FLAGS.num_epochs)
        train_batches = data.train_batch_iter()

        # loading devolopment data
        dev_x, dev_y = data.get_dev_data()

        # declare model
        model = TextCNN(forced_seq_len=data.forced_seq_len,
                        vocab_size=data.vocab_size,
                        embedding_size=FLAGS.embedding_size,
                        filters_size_list=FLAGS.filters_size_list,
                        num_classes=FLAGS.num_classes,
                        num_filters=FLAGS.num_filters)

        # define training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        train_op = optimizer.minimize(
            loss=model.loss_op, global_step=global_step)

        # initialize all variables
        sess.run(tf.global_variables_initializer())

        # training and eval into dev set
        while 1:
            batch_x, batch_y = next(train_batches)
            _, step = sess.run([train_op, global_step], feed_dict={
                               model.inputs: batch_x, model.labels: batch_y, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
            if step % FLAGS.evaluate_every == 0:
                loss, acc = sess.run([model.loss_op, model.accuracy_op], feed_dict={
                                     model.inputs: dev_x, model.labels: dev_y, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
                print('setp:{} | loss:{} | acc:{}'.format(step, loss, acc))
