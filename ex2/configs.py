INIT_SEED = 42

# MODEL_TYPES = ['LSTM', 'GRU']
MODEL_TYPES = ['LSTM']

# DROPOUT_VALS = [0.0, 0.5]
# DROPOUT_VALS = [0.1, 0.25, 0.5, 0.75]
DROPOUT_VALS = [0.25]

OUTPUTS_DIR = 'results'
MODELS_OUTPUT_DIR = 'results/models'
TENSORBOARD_DIR = 'results/tensor_board'

TRAIN_DATA = "data/PTB/ptb.train.txt"
VAL_DATA = "data/PTB/ptb.valid.txt"
TEST_DATA = "data/PTB/ptb.test.txt"

WORD_PRETRAINED_EMBED_FILE = "data/GloVe/glove.6B.200d.txt"
