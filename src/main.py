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
from settings import TrainingSettings, TestSettings, InferenceSettings, ValidationError, Settings, Command


class App:

    def __init__(self):
        self.settings: TrainingSettings | TestSettings | InferenceSettings | None = None
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.dataloader_train, self.dataloader_test, self.dataloader_validate = None, None, None

    def _init_dataset(self):
        dataset = DataSet(
            self.settings.path_to_data,
            self.settings.dcm_dir,
            self.settings.png_dir,
            self.settings.img_size,
            self.settings.use_augmentation,
            self.settings.seed
        )
        ds_train, ds_test, ds_val = torch.utils.data.random_split(dataset, 
            [
                self.settings.train_set, 
                self.settings.test_set,
                self.settings.val_set
            ]
        )
        self.dataloader_train = DataLoader(ds_train, batch_size=self.settings.batch_size, num_workers=1, shuffle=True, drop_last=True)
        self.dataloader_test = DataLoader(ds_test, batch_size=self.settings.batch_size, num_workers=1, shuffle=True, drop_last=True)
        self.dataloader_validate = DataLoader(ds_val, batch_size=self.settings.batch_size, num_workers=1, shuffle=True, drop_last=True)

    def _init_model(self):
        self.model = Unet(
            backbone_name=self.settings.backbone,
            pretrained=True,
            classes=1
        )
        if os.path.exists(self.settings.save_path):
            state_dict = torch.load(self.settings.save_path, map_location=torch.device('cpu'))
            self.model.load_state_dict(state_dict)
        self.model.to(self.device)

    @property
    def is_dataset_inited(self):
        return self.dataloader_test is not None and self.dataloader_train is not None and self.dataloader_validate is not None

    @staticmethod
    def exit():
        sys.exit()

    def train(self):
        if not self.is_dataset_inited:
            self._init_dataset()
        if self.model is None:
            self._init_model()
        Trainer(
            model=self.model, 
            device=self.device, 
            train_dataloader=self.dataloader_train,
            val_dataloader=self.dataloader_validate,
            epochs=self.settings.epochs,
            save_path=self.settings.save_path,
            lr=self.settings.lr
        ).run()

    def test(self):
        if not self.is_dataset_inited:
            self._init_dataset()
        if self.model is None:
            self._init_model()
        prec, recall, f1, jac, dice = Tester(
            model=self.model,
            device=self.device,
            dataloader=self.dataloader_test,
            batch_size=self.settings.batch_size
        ).run()
        print(f'''
Precision: {prec}
Recall: {recall}
F1: {f1}
Jaccard Coefficient: {jac}
Dice Coefficient: {dice}
''')

    def inference(self):
        if self.model is None:
            self._init_model()
        self.model.eval()
        if os.path.isdir(self.settings.input_path):
            for file in glob(f'{self.settings.input_path}/*.dcm'):
                self._inference_file(file)
        else:
            self._inference_file(self.settings.input_path)

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
        if not os.path.exists(self.settings.output_path):
            os.mkdir(self.settings.output_path)
        torchvision.utils.save_image(pred, f"{self.settings.output_path}/{path.split('/')[-1].replace('.dcm', '.png')}")

    def run(self):
        args = sys.argv[1::]
        if len(args) < 1 or args[0] not in Command.values():
            print('Please provide correct subcommand.\n')
            sys.argv = [sys.argv[0]] + ['-h']
        try:
            settings = Settings()
        except ValidationError:
            print('Invalid input! Check if you provide all required args according to their types!')
            self.exit()
        except Exception:
            print('Some error occurred! Please check provided args values.')
            sys.exit()
        self.settings = settings.current
        if not isinstance(self.settings, InferenceSettings):
            self.settings.validate_sub_paths()
        task = {
            Command.train: self.train,
            Command.test: self.test,
            Command.inference: self.inference
        }[settings.command]
        task()


def run_cli():
    App().run()
