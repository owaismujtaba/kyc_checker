import os

CUR_DIR = os.getcwd()

DATA_FOLDER = os.path.join(CUR_DIR, 'Data')
TRAIN_SAMPLE_FOLDER = os.path.join(DATA_FOLDER, 'train_sample_videos')
TEST_FOLDER = os.path.join(DATA_FOLDER, 'test_videos')
IMG_SIZE = 650

MAX_FRAMES = 2
BATCH_SIZE = 16
EPOCHS = 1

MODEL_PATH = os.path.join(CUR_DIR, 'artifacts', 'models','fake_image_detector.h5')
THRESHOLD = 0.5
OUTPUT_CLASSES = 1