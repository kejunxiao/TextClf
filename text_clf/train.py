import tensorflow as tf
from data_load import DataLoad
from model import TextCNN

# Parameters
# Data loading parameters
tf.app.flags.DEFINE_float("dev_sample_rate", .2,
                          "Percentage of the training data to use for validation")
tf.app.flags.DEFINE_string("train_data_path",
                           "../dataset/San_Francisco_Crime/train.csv.zip", "Data source for the train data.")

# Model Hyperparameters
tf.app.flags.DEFINE_float("learning_rate", 1e-3,
                          "learning rate (default:0.001)")
tf.app.flags.DEFINE_integer("embedding_size", 128,
                            "Dimensionality of character embedding (default: 128)")
tf.app.flags.DEFINE_list("filters_size_list", [
                         3, 4, 5], "list type filter sizes (default: [3, 4, 5])")
tf.app.flags.DEFINE_integer("num_filters", 4,
                            "Number of filters per filter size (default: 128)")
tf.app.flags.DEFINE_float("dropout_keep_prob", .5,
                          "Dropout keep probability (default: 0.5)")
tf.app.flags.DEFINE_integer("num_classes", 39, "number of classes (default: 39)")

# Training schemes
tf.app.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.app.flags.DEFINE_integer("num_epochs", 200,
                            "Number of training epochs (default: 200)")
# tf.app.flags.DEFINE_integer("evaluate_every", 100,
#                             "Evaluate model on dev set after this many steps (default: 100)")
# tf.app.flags.DEFINE_integer("checkpoint_every", 100,
#                             "Save model after this many steps (default: 100)")
# tf.app.flags.DEFINE_integer("num_checkpoints", 5,
#                             "Number of checkpoints to store (default: 5)")

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

        train_set = DataLoad(data_path=FLAGS.train_data_path,
                             mode='train',
                             batch_size=FLAGS.batch_size,
                             num_epochs=FLAGS.num_epochs,
                             dev_sample_rate=FLAGS.dev_sample_rate)
        batches = train_set.batch_iter()

        model = TextCNN(forced_seq_len=train_set.forced_seq_len,
                        vocab_size=train_set.vocab_size,
                        embedding_size=FLAGS.embedding_size,
                        filters_size_list=FLAGS.filters_size_list,
                        num_classes=FLAGS.num_classes,
                        num_filters=FLAGS.num_filters)

        # define training procedure
        global_step = tf.Variable(0, trainable=False, name='global_step')
        optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(loss=model.loss_op)
        train_op = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # initialize all variables
        sess.run(tf.global_variables_initializer())

        for batch_x, batch_y in batches:
            _, steps, loss, acc = sess.run(
                [train_op, global_step, model.loss_op, model.accuracy_op])
            print('setps:{} | loss:{} | acc:{}'.format(steps, loss, acc))
