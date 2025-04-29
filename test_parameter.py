FOLDER_NAME = 'ariadne1'
model_path = f'model/{FOLDER_NAME}'
test_gifs_path = f'gifs/{FOLDER_NAME}/test'
test_trajectory_path = f'results/trajectory'
test_length_path = f'results/length'

NODE_INPUT_DIM = 4
EMBEDDING_DIM = 128
K_SIZE = 25

USE_GPU = True
NUM_GPU = 1
NUM_META_AGENT = 2
NUM_TEST = 5
NUM_RUN = 3
SAVE_GIFS = True
SAVE_TRAJECTORY = True
SAVE_LENGTH = True
MAX_EPISODE_STEP = 128
FREE = 255