import os

ROOT_PATH = os.path.dirname(os.getcwd())
DATA_PATH = os.path.join(ROOT_PATH, "data")
TRAIN_ORIGINAL_FILE_PATH = os.path.join(DATA_PATH, "train_original.pkl")
TRAIN_ROTATED_FILE_PATH = os.path.join(DATA_PATH, "train_rotated.pkl")
TEST_ORIGINAL_FILE_PATH = os.path.join(DATA_PATH, "test_original.pkl")
TEST_ROTATED_FILE_PATH = os.path.join(DATA_PATH, "test_rotated.pkl")
