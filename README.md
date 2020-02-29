Academic project aiming at creating a efficient deep learning module to train models on classification problems.

# How to use the code

Parametrisation of the training has to be changed in the file `config.py` and to train the model run `main.py`

# Training data formatting

The training data location has to be defined in the configuration : in the global variable `DATA_STUDIED` 
The data need to be organized in folders. One folder for each class. Following the case the configuration variable `SPLIT` must take one specific mode. If the data are :
- already split in a test and train folder use the mode `SplitOptions.SPLIT_TRAIN`
- already split in a test, train and validation folder use the mode `SplitOptions.NO_SPLIT`
- only in one folder, use the option `SplitOptions.SPLIT_ALL`

# Modules details

## Data-saver
A python module to save the data of a training : 
 - parameters for the training (number of épochs, name of the model, batch size, learning rate ...)
 - hyperparameters of the models chosen for the training (height of layers, size of kernels ...)
 - (val accuracy, loss train, loss val, confusion matrix) for each épochs
 - accuracy, confusion matrix and loss on the test dataset
 
 ## Learning
 The module for the training of models
 
 ## Scripts
 A module with a script to format incoming datasets (from csv to folder architecture)
 
 ## Common 
 A module with some helpful functions and enums
