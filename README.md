# LUNG INJURY DETECTION CLI APP

___

This repository provides source code of CLI app for training and inferring lung injury detection model. The model is based on U-Net architecture implemented with pytorch framework for python. 

## INSTALLATION

___

<u>The project depends on python ==3.10.*</u>

Install from PyPI:
```shell
pip install LungDamageDetectionCLI
```
Install from repo:
```shell
git clone <repo-name> && cd <project-dir>
pdm install
```

## DEVELOPMENT AND USE

___

<u>Copy .env.example and rename it into .env</u>

<b>Environment Variables</b>

    PATH_TO_DATA=  # directory containing dataset 
    DCM_DIR=  # directory inside PATH_TO_DATA containing dcm files
    PNG_DIR=  # directory inside PATH_TO_DATA containing png files
    TRAIN_SET=  # number of images for model training
    TEST_SET=  # number of images for model test
    VAL_SET=  # number of images for model validation
    N_EPOCHS=200 # number of training epochs
    LR=1e-4 # learning rate
    BATCH_SIZE=4 # size of batch
    IMG_SIZE=512 # width and height of images in dataset
    BACKBONE=resnet101 # name of backbone model
    SAVE_PATH=  # path of trained model state dict
    OUT_PATH=  # directory to save inference results
    USE_AUGMENTATION=false # dataset is loaded with augmentation if true
    SEED=5 # seed for numpy.random.seed

Supported backbone names:
* resnet18
* resnet34
* resnet50
* resnet101 
* resnet152
* vgg16
* vgg19
* densenet121
* densenet161
* densenet169
* densenet201

Run CLI app from project dir
```shell
pdm run ldd-cli
```

This app supports these commands:
* help - print list of supported commands
* exit - close app
* train - run model training
* test - run model test
* inference - detect injuries (only dcm files)

All code is placed in src directory.

To train model place .dcm files of CT of lungs in PATH_TO_DATA/DCM_DATA_DIR and .png masks in PATH_TO_DATA/PNG_DATA_DIR then set TRAIN_SET, VAL_SET and TEST_SET variables according to your dataset size. App saves state dict of model every epoch. If SAVE_PATH file exists model will be loaded using it when the app is restarted.

If USE_AUGMENTATION is true some images will be rotated, cropped and flipped.

After model test app prints metrics of trained model.

<b>Metrics</b>
- precision
- recall
- f1 score
- jaccard coefficient
- dice coefficient

To inference model you have to provide path of dcm file or path of directory with several dcm files.

Result png files are placed in OUT_PATH. Png files have same name as dcms.

## LICENSE

___

Lung injury detection CLI app is [MIT licensed](LICENSE)
