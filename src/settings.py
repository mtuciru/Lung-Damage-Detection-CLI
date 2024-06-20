import os
import sys
from dotenv import load_dotenv


load_dotenv()


def get_env(var):
    val = os.environ.get(var)
    if val is None:
        print(f'{var} is not provided! Check .env file.')
        sys.exit()
    return val

def get_path_env(var, sub='.'):
    path = get_env(var)
    if not os.path.exists(f'{sub}/{path}'):
        print(f'{sub}/{path} does not exists! Check {var} variable.')
        sys.exit()
    return path


def get_int_env(var):
    n = get_env(var)
    try:
        return int(n)
    except ValueError:
        print(f'Invalid {var} value! Provide int number.')
        sys.exit()


def get_float_env(var):
    n = get_env(var)
    try:
        return float(n)
    except ValueError:
        print(f'Invalid {var} value! Provide float number.')
        sys.exit()


def get_backbone(var):
    backbone = get_env(var)
    if backbone not in (
        'resnet18',
        'resnet34',
        'resnet50',
        'resnet101', 
        'resnet152',
        'vgg16',
        'vgg19',
        'densenet121',
        'densenet161',
        'densenet169',
        'densenet201'
    ):
        print('Invalid backbone name! Check .env file. Plaese check list of supported values in README.')
        sys.exit()
    return backbone


PATH_TO_DATA = get_path_env('PATH_TO_DATA')
DCM_DIR = get_path_env('DCM_DIR', sub=PATH_TO_DATA)
PNG_DIR = get_path_env('PNG_DIR', sub=PATH_TO_DATA)

TRAIN_SET = get_int_env('TRAIN_SET')
TEST_SET = get_int_env('TEST_SET')
VAL_SET = get_int_env('VAL_SET')

SAVE_PATH = get_env('SAVE_PATH')
OUT_PATH = get_env('OUT_PATH')

N_EPOCHS = get_int_env('N_EPOCHS')
BATCH_SIZE = get_int_env('BATCH_SIZE')

LR = get_float_env('LR')
IMG_SIZE = get_int_env('IMG_SIZE')
BACKBONE = get_backbone('BACKBONE')

USE_AUGMENTATION = get_env('USE_AUGMENTATION').lower() in ('true', '1')
SEED = get_int_env('SEED')
