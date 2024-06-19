import os
from dotenv import load_dotenv


load_dotenv()

PATH_TO_DATA = os.environ.get('PATH_TO_DATA')
DCM_DIR = os.environ.get('DCM_DIR')
PNG_DIR = os.environ.get('PNG_DIR')

TRAIN_SET = int(os.environ.get('TRAIN_SET'))
TEST_SET = int(os.environ.get('TEST_SET'))
VAL_SET = int(os.environ.get('VAL_SET'))
SAVE_PATH = os.environ.get('SAVE_PATH')
OUT_PATH = os.environ.get('OUT_PATH')

N_EPOCHS = int(os.environ.get('N_EPOCHS'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE'))

LR = float(os.environ.get('LR'))
IMG_SIZE = int(os.environ.get('IMG_SIZE'))
BACKBONE = os.environ.get('BACKBONE')

USE_AUGMENTATION = str(os.getenv('USE_AUGMENTATION')).lower() in ('true', '1')
SEED = int(os.environ.get('SEED'))
