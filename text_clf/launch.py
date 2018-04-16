import tensorflow as tf
from train import train
from evaluate import evaluate

if __name__ == '__main__':
    # Parameters
    # Data loading parameters
    tf.app.flags.DEFINE_float("dev_sample_rate", .05,
                              "Percentage of the training data to use for validation(default:0.05)")
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
    tf.app.flags.DEFINE_integer("num_filters", 128,
                                "Number of filters per filter size (default: 128)")
    tf.app.flags.DEFINE_float("dropout_keep_prob", .5,
                              "Dropout keep probability (default: 0.5)")
    tf.app.flags.DEFINE_integer("num_classes", 39,
                                "number of classes (default: 39)")

    # Training schemes
    tf.app.flags.DEFINE_boolean("is_training", True, 
                                "if True , the mode is training, False is eval(default:True")
    tf.app.flags.DEFINE_integer("batch_size", 64, 
                                "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", 200, 
                            "Number of training epochs (default: 200)")
    tf.app.flags.DEFINE_integer("max_to_keep", 5,
                                "tf.train.Saver(max_to_keep) (default:5)")
    tf.app.flags.DEFINE_integer("evaluate_every", 100,
                                "Evaluate model on dev set after this many steps (default: 100)")

    # Misc Parameters
    tf.app.flags.DEFINE_boolean("allow_soft_placement", True,
                                "Allow device soft device placement")
    tf.app.flags.DEFINE_boolean("log_device_placement", False,
                                "Log placement of ops on devices")

    FLAGS = tf.app.flags.FLAGS

    if FLAGS.is_training:
        train(FLAGS)
    elif not FLAGS.is_training:
        evaluate(FLAGS)
