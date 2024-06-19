from glob import glob

import numpy as np
import pydicom as dicom
import torch
import torchvision
from torch.utils.data import Dataset

from settings import IMG_SIZE, DCM_DIR, PNG_DIR, PATH_TO_DATA, USE_AUGMENTATION, SEED


np.random.seed(SEED)


class DataSet(Dataset):

    def __init__(self):
        self.paths_x = sorted(glob(f'{PATH_TO_DATA}/{DCM_DIR}/*.dcm'))
        self.paths_y = sorted(glob(f'{PATH_TO_DATA}/{PNG_DIR}/*.png'))
        self.use_augmentation = USE_AUGMENTATION

    def __len__(self):
        return len(self.paths_x)

    def __getitem__(self, item):
        x, y = self.paths_x[item], self.paths_y[item]

        x, y = self.__load_file(x, y)

        if self.use_augmentation:
            x, y = self.__augmentation(x, y)

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

    @staticmethod
    def __normalize(arr, is_dcm):
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
            arr = torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE])(arr)
        return arr

    @staticmethod
    def __augmentation(x, y):

        if np.random.random() > 0.5:
            x = torch.fliplr(x)
            y = torch.fliplr(y)

        if np.random.random() > 0.5 or True:
            choice = np.random.randint(1, 3)
            x = torchvision.transforms.functional.rotate(x, choice * 90)
            y = torchvision.transforms.functional.rotate(x, choice * 90)

        if np.random.random() > 0.5 or True:
            x = torchvision.transforms.RandomCrop(size=[IMG_SIZE // 2, IMG_SIZE // 2])(x)
            y = torchvision.transforms.RandomCrop(size=[IMG_SIZE // 2, IMG_SIZE // 2])(y)

            x = torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE])(x)
            y = torchvision.transforms.Resize([IMG_SIZE, IMG_SIZE])(y)

        x = torchvision.transforms.ColorJitter(brightness=0.2, hue=0.2)(x)

        return x, torchvision.transforms.Grayscale()(y)
