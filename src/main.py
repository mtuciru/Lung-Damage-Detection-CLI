import os
import sys
import warnings
warnings.filterwarnings('ignore')

from glob import glob
from enum import Enum

import pydicom as dicom
import torch
from torch.utils.data import DataLoader
import torchvision

from model import Unet
from data import DataSet
from trainer import Trainer
from tester import Tester
from settings import (
    TRAIN_SET, 
    TEST_SET, 
    VAL_SET, 
    BATCH_SIZE,
    SAVE_PATH,
    BACKBONE,
    OUT_PATH
)


class Command(Enum):
    train = 'train'
    test = 'test'
    inference = 'inference'
    exit = 'exit'
    help = 'help'

    @staticmethod
    def values():
        return [c.value for c in Command]


class App:

    def __init__(self):
        self.is_dataset_inited = False
        self.model = Unet(
            backbone_name=BACKBONE,
            pretrained=True,
            classes=1
        )
        if os.path.exists(SAVE_PATH):
            state_dict = torch.load(SAVE_PATH, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.dataloader_train, self.dataloader_test, self.dataloader_validate = None, None, None

    def _init_dataset(self):
        dataset = DataSet()
        ds_train, ds_test, ds_val = torch.utils.data.random_split(dataset, [TRAIN_SET, TEST_SET, VAL_SET])
        self.dataloader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
        self.dataloader_test = DataLoader(ds_test, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
        self.dataloader_validate = DataLoader(ds_val, batch_size=BATCH_SIZE, num_workers=1, shuffle=True, drop_last=True)
        self.is_dataset_inited = True

    def help(self):
        print('''
Lung injury detection cli app.
              
Commands:
* help - print this message
* exit - close app
* train - run model training
* test - run model test
* inference - detect injuries (only dcm files)
''')

    @staticmethod
    def read_command():
        while True:
            command = input('>>> ').lower()
            if command in Command.values():
                return Command(command)
            print('\nInvalid Command!\n')
            continue
    
    @staticmethod
    def read_file_or_dir():
        while True:
            path = input('Input dcm file or directory path. Press enter to return: ')
            if not path:
                return
            elif not os.path.exists(path):
                print('\nThis path does not exists!\n')
            elif not (is_dir := os.path.isdir(path)) and path[-4::] != '.dcm':
                print('\nInvalid file! Use only dcm files.\n')
            else:
                return path, is_dir

    def train(self):
        if not self.is_dataset_inited:
            self._init_dataset()
        Trainer(
            model=self.model, 
            device=self.device, 
            train_dataloader=self.dataloader_train,
            val_dataloader=self.dataloader_validate
        ).run()

    def test(self):
        if not self.is_dataset_inited:
            self._init_dataset()
        prec, recall, f1, jac, dice = Tester(
            model=self.model,
            device=self.device,
            dataloader=self.dataloader_test
        ).run()
        print(f'''
Precision: {prec}
Recall: {recall}
F1: {f1}
Jaccard Coefficient: {jac}
Dice Coefficient: {dice}
''')

    def inference(self):
        self.model.eval()
        path_info = self.read_file_or_dir()
        if path_info is None:
            return
        path, is_dir = path_info
        if is_dir:
            for file in glob(f'{path}/*.dcm'):
                self._inference_file(file)
        else:
            self._inference_file(path)
    
    def _inference_file(self, path):
        file = dicom.read_file(path).pixel_array
        arr = torch.from_numpy(file)
        arr = torch.unsqueeze(arr, dim=0)
        arr[arr < -1024.0] = 0.0
        arr = (arr - -1000) / (500 - -1000)
        arr[arr > 1.0] = 1.
        arr[arr < 0.0] = 0.
        arr = arr.repeat(3, 1, 1)
        arr = arr.unsqueeze(0)
        arr.to(self.device)
        arr = arr.to(self.device)
        pred = self.model(arr)
        pred = torch.where(pred > 0.2, 1.0, 0.0)
        if not os.path.exists(OUT_PATH):
            os.mkdir(OUT_PATH)
        torchvision.utils.save_image(pred, f"{OUT_PATH}/{path.split('/')[-1].replace('.dcm', '.png')}")

    def run(self):
        self.help()
        while True:
            command = self.read_command()
            {
                Command.exit: sys.exit,
                Command.train: self.train,
                Command.test: self.test,
                Command.inference: self.inference,
                Command.help: self.help
            }[command]()


def run_cli():
    App().run()
