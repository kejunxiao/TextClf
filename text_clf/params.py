"""
parameters for testing
"""
# data
DATA_PATH = '../dataset/San_Francisco_Crime/train.csv.zip'
VOCAB_SIZE = None
FORCED_SEQ_LEN = 14
NUM_CLASSES = 39

# training scheme
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUM_EPOCHS = 10000

# graph params
EMBED_SIZE = 256
DROPOUT_KEEP_PROB = 0.5
FILTERS_SIZE_LIST = [3, 4, 5]
NUM_FILTERS = 32