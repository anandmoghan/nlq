LOG_FILE = 'LogFile-{%}.log'
BATCH_SIZE = 512

# Glove and Embedding Constants
GLOVE_TOKENS = 6  # Billion
EMBEDDING_SIZE = 300  # Dimensions
GLOVE = '../glove/glove.{}B.{}d.txt'.format(GLOVE_TOKENS, EMBEDDING_SIZE)
GLOVE_SAVE = '../glove/loaded.pkl'
TOKEN_TO_IDX_SAVE = '../data/token_to_index.pkl'
TOKEN_WEIGHTS_SAVE = '../data/token_weights.pkl'
UNK_TOKEN = '<UNK>'
BEG_TOKEN = '<BEG'
END_TOKEN = '<END>'
UNK_IDX = 0
BEG_IDX = 1
END_IDX = 2

MAX_LEN = 76

# Data Constants
DATA_DIR = '../data/'

# Debug Constants
DEBUG_DATA_SIZE = 1000
DEBUG_BATCH_SIZE = 16

# Model Constants
EARLY_STOP_COUNT = 3

TRAIN_ALL = [True, True, True]
TRAIN_AGG = [True, False, False]
TRAIN_SEL = [False, True, False]
TRAIN_WHERE = [False, False, True]


AGG_LR = 0.01
AGG_DECAY = 0.90
AGG_COUNT = 6

AGG_RNN_LAYERS = 2
AGG_RNN_SIZE = 128

SEL_LR = 0.001
SEL_DECAY = 0.95
SEL_COUNT = 22

SEL_RNN_LAYERS = 2
SEL_RNN_SIZE = 256

WHERE_COL_COUNT = 43
WHERE_OP_COUNT = 10
WHERE_LR = 0.01
WHERE_DECAY = 0.90

WHERE_RNN_LAYERS = 2
WHERE_RNN_SIZE = 256


