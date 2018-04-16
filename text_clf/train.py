import os
import time
import tensorflow as tf
from data_load import DataLoad
from model import TextCNN


def train(FLAGS):
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
                            num_epochs=FLAGS.num_epochs,
                            dev_sample_rate=FLAGS.dev_sample_rate)
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
            optimizer = tf.train.AdamOptimizer(
                learning_rate=FLAGS.learning_rate)
            train_op = optimizer.minimize(
                loss=model.loss_op, global_step=global_step)

            # tensorboard graph visualizing
            graph_writer = tf.summary.FileWriter(logdir='../graph/model/')
            graph_writer.add_graph(sess.graph)

            # summaries director
            summary_dir = '../summaries/'
            loss_summary = tf.summary.scalar('loss_summaries', model.loss_op)
            acc_summary = tf.summary.scalar('accuracy', model.accuracy_op)

            train_summary_dir = os.path.join(summary_dir, 'train')
            dev_summary_dir = os.path.join(summary_dir, 'dev')

            train_merged_summaries = tf.summary.merge(
                [loss_summary, acc_summary])
            dev_merged_summaries = tf.summary.merge(
                [loss_summary, acc_summary])

            train_summaries_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
            dev_summaries_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # initialize all variables
            sess.run(tf.global_variables_initializer())

            # training and eval into dev set
            while 1:
                batch_x, batch_y = next(train_batches)
                _, step, loss, acc, summaries = sess.run(
                    [train_op, global_step, model.loss_op, model.accuracy_op, train_merged_summaries], feed_dict={
                    model.inputs: batch_x, model.labels: batch_y, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
                train_summaries_writer.add_summary(summaries, step)
                if step % FLAGS.evaluate_every == 0:
                    loss, acc, summaries = sess.run([model.loss_op, model.accuracy_op, dev_merged_summaries], feed_dict={
                        model.inputs: dev_x, model.labels: dev_y, model.dropout_keep_prob: FLAGS.dropout_keep_prob})
                    print('setp:{} | loss:{} | acc:{}'.format(step, loss, acc))
                    dev_summaries_writer.add_summary(summaries, step)
