# FOLDER_NAME = 'ae_clean'
FOLDER_NAME = 'ae_clean_229 '
model_path = f'model/{FOLDER_NAME}'
train_path = f'train/{FOLDER_NAME}'
gifs_path = f'gifs/{FOLDER_NAME}'

SUMMARY_WINDOW = 32
LOAD_MODEL = False  # do you want to load the model trained before
SAVE_IMG_GAP = 100

INPUT_DIM = 4
EMBEDDING_DIM = 128
NODE_PADDING_SIZE = 360  # the number of nodes will be padded to this value
K_SIZE = 20  # the number of neighboring nodes

# USE_GPU = False  # do you want to collect training data using GPUs
USE_GPU = True  # do you want to collect training data using GPUs
USE_GPU_GLOBAL = True  # do you want to train the network using GPUs
NUM_GPU = 1
NUM_META_AGENT = 2

REPLAY_SIZE = 10000
MINIMUM_BUFFER_SIZE = 2000
BATCH_SIZE = 4
LR = 1e-5
GAMMA = 1
DECAY_STEP = 256  # not use
