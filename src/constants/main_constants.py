LOG_FILE = 'LogFile-{%}.log'
BATCH_SIZE = 64

# Glove and Embedding Constants
GLOVE_TOKENS = 6  # Billion
EMBEDDING_SIZE = 50  # Dimensions
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
LEARNING_RATE = 0.001

AGG_COUNT = 6


AGG_EMB_SAVE_MODEL = '../save/agg_emb_model_accuracy_{:.2f}'
AGG_SAVE_MODEL = '../save/agg_model_accuracy_{:.2f}'
# Aggregate Predictor Parameters
AGG_GRAD_CLIP = 0.25
AGG_CNN_1_NUM_FILTERS = 128  # Number of filters should be equal to embedding size
AGG_CNN_1_KERNEL_SIZE = (3, EMBEDDING_SIZE)  # Kernel Width should be equal to embedding size
AGG_CNN_1_STRIDE = 1

AGG_CNN_2_NUM_FILTERS = 64
AGG_CNN_2_KERNEL_SIZE = (MAX_LEN, 1)
AGG_CNN_2_STRIDE = 1

AGG_RNN_LAYERS = 2
AGG_RNN_SIZE = 128

AGG_CNN_DROPOUT = .3
AGG_RNN_DROPOUT = .7

