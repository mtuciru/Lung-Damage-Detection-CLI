from glob import glob

import numpy as np
import pydicom as dicom
import torch
import torchvision
from torch.utils.data import Dataset


class DataSet(Dataset):

    def __init__(
            self, 
            path_to_data: str, 
            dcm_dir: str,
            png_dir: str,
            img_size: int,
            use_aug: bool,
            seed: int
    ):
        self.paths_x = sorted(glob(f'{path_to_data}/{dcm_dir}/*.dcm'))
        self.paths_y = sorted(glob(f'{path_to_data}/{png_dir}/*.png'))
        self.use_augmentation = use_aug
        if self.use_augmentation:
            np.random.seed(seed)
        self.img_size = img_size

    def __len__(self):
        return len(self.paths_x)

    def __getitem__(self, item):
        x, y = self.paths_x[item], self.paths_y[item]

        x, y = self.__load_file(x, y)

        if self.use_augmentation:
            x, y = self.__augmentation(x, y)
            y = self.__normalize(y, False)

        x, y = x.float(), y.float()

        return x, y

    @staticmethod
    def __load(path, is_dcm):
        if is_dcm:
            file = dicom.read_file(path).pixel_array
            file = torch.from_numpy(file)
        else:
            file = torchvision.io.read_image(path, mode=torchvision.io.image.ImageReadMode.GRAY)
        return file

    def __load_file(self, x, y):
        x, y = self.__load(x, is_dcm=True), self.__load(y, is_dcm=False)
        x, y = self.__normalize(x, is_dcm=True), self.__normalize(y, is_dcm=False)
        return x, y

    def __normalize(self, arr, is_dcm):
        if is_dcm:
            arr = torch.unsqueeze(arr, dim=0)
            arr[arr < -1024.0] = 0.0
            arr = (arr - -1000) / (500 - -1000)
            arr[arr > 1.0] = 1.
            arr[arr < 0.0] = 0.
            arr = arr.repeat(3, 1, 1)
        else:
            arr = arr.float()
            arr = arr / 255.0
            arr = torch.where(arr > 0, 1.0, 0.0)
            arr = torchvision.transforms.Resize([self.img_size, self.img_size])(arr)
        return arr

    def __augmentation(self, x, y):

        if np.random.random() > 0.5:
            x = torch.fliplr(x)
            y = torch.fliplr(y)

        if np.random.random() > 0.5 or True:
            choice = np.random.randint(1, 3)
            x = torchvision.transforms.functional.rotate(x, choice * 90)
            y = torchvision.transforms.functional.rotate(x, choice * 90)

        if np.random.random() > 0.5 or True:
            x = torchvision.transforms.RandomCrop(size=[self.img_size // 2, self.img_size // 2])(x)
            y = torchvision.transforms.RandomCrop(size=[self.img_size // 2, self.img_size // 2])(y)

            x = torchvision.transforms.Resize([self.img_size, self.img_size])(x)
            y = torchvision.transforms.Resize([self.img_size, self.img_size])(y)

        x = torchvision.transforms.ColorJitter(brightness=0.2, hue=0.2)(x)

        return x, torchvision.transforms.Grayscale()(y)
