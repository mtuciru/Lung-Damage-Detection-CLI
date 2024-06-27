import os
import sys
from enum import Enum

from pydantic import field_validator, ValidationError, Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict, CliSubCommand


class Command(Enum):
    train = 'train'
    test = 'test'
    inference = 'inference'

    @staticmethod
    def values():
        return [c.value for c in Command]


class _Base(BaseModel):

    @field_validator(
        'train_set', 
        'test_set', 
        'val_set', 
        'epochs', 
        'seed', 
        'batch_size', 
        'img_size', 
        check_fields=False, 
        mode='before'
    )
    @classmethod
    def validate_int(cls, n):
        try:
            n = int(n)
        except ValidationError:
            print('Invalid args! Please check if you are trying to pass string into integer args!')
        else:
            if n >= 0:
                return n
            else:
                print('Invalid args! Please check if you provide non negative numbers into int args!')
        sys.exit()

    @field_validator(
        'lr', 
        check_fields=False, 
        mode='before'
    )
    @classmethod
    def validate_float(cls, n):
        try:
            return float(n)
        except ValidationError:
            print('Invalid args! Please check if you are trying to pass string into float args!')
            sys.exit()

    @field_validator(
        'path_to_data', 
        check_fields=False
    )
    @classmethod
    def validate_dir_path(cls, path):
        if os.path.exists(path) and os.path.isdir(path):
            return path
        elif not os.path.exists(path):
            print(f'"{path}" does not exist!')
        else:
            print(f'"{path}" is not a directory!')            
        sys.exit()

    @classmethod
    def validate_file_path(cls, path):
        if os.path.exists(path) and not os.path.isdir(path):
            return path
        elif not os.path.exists(path):
            print(f'"{path}" does not exist!')
        else:
            print(f'"{path}" is a directory!')            
        sys.exit()


class DatasetArgs(_Base):

    path_to_data: str = Field(description='directory containing dataset')
    dcm_dir: str = Field(description='directory inside PATH_TO_DATA containing dcm files')
    png_dir: str = Field(description='directory inside PATH_TO_DATA containing png files')

    train_set: int = Field(description='number of images for model training')
    test_set: int = Field(description='number of images for model test')
    val_set: int = Field(description='number of images for model validation')

    batch_size: int = Field(description='size of batch')
    img_size: int = Field(default=512, description='width and height of images in dataset')

    use_augmentation: bool = Field(default=False, description='dataset is loaded with augmentation if true')
    seed: int = Field(default=5, description='seed for numpy.random.seed')

    @field_validator(
        'path_to_data', 
        'dcm_dir', 
        'png_dir', 
        'train_set', 
        'test_set', 
        'val_set', 
        'batch_size', 
        'img_size', 
        'use_augmentation', 
        'seed', 
        mode='before'
    )
    @classmethod
    def check_if_provided(cls, n):
        if n is None:
            print('Invalid input! Check if you privide all required args!')
            sys.exit()
        return n

    def validate_sub_paths(self):
        for sub in (self.dcm_dir, self.png_dir):
            _path = f'{self.path_to_data}/{sub}'
            if os.path.exists(_path) and os.path.isdir(_path):
                return sub
            elif not os.path.exists(_path):
                print(f'{_path} does not exist!')
            else:
                print(f'{_path} is not a directory!')  
            sys.exit()


class ModelArgs(_Base):

    save_path: str = Field(description='path of trained model state dict')
    backbone: str = Field(description='name of backbone model')

    @field_validator('backbone')
    @classmethod
    def validate_backbone(cls, backbone):
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
            print('Invalid backbone name! Plaese check list of supported values in README.')
            sys.exit()
        return backbone


class _BaseSettings(BaseSettings):

    model_config = SettingsConfigDict(env_file='.env', extra='ignore')


class InferenceSettings(_BaseSettings, ModelArgs):

    output_path: str = Field(description='directory to save inference results')
    input_path: str = Field(description='dcm file or directory of dcm files for model inference')

    @classmethod
    def validate_dcm(cls, path):
        if path[-4::] != '.dcm':
            print('Invalid args! Please provide only dcm files for inference!')
            sys.exit()
        return path

    @field_validator('input_path')
    @classmethod
    def validate_input_path(cls, path):
        if not os.path.exists(path):
            print(f'{path} does not exist!')
            sys.exit()
        if os.path.isdir(path):
            return cls.validate_dir_path(path)
        else:
            path = cls.validate_dcm(path)
            return cls.validate_file_path(path)


class TrainingSettings(_BaseSettings, ModelArgs, DatasetArgs):

    epochs: int = Field(description='number of training epochs')
    lr: float = Field(description='learning rate')


class TestSettings(_BaseSettings, ModelArgs, DatasetArgs):

    pass


class Settings(BaseSettings, cli_parse_args=True, cli_prog_name='Lung Damage Detection', extra='allow'):

    train: CliSubCommand[TrainingSettings] = Field(description='train model and save state dict')
    test: CliSubCommand[TestSettings] = Field(description='test trained model')
    inference: CliSubCommand[InferenceSettings] = Field(description='inference trained model')

    @property
    def current(self):
        try:
            return {
                self.train is not None: self.train,
                self.test is not None: self.test,
                self.inference is not None: self.inference
            }[True]
        except Exception:
            print('Invalid input! Plaease check provided args!')
            sys.exit()

    @property
    def command(self):
        return {
            self.train is not None: Command.train,
            self.test is not None: Command.test,
            self.inference is not None: Command.inference
        }[True]
