from common.data_imports import SplitOptions
from common.helper import ModelEnum, DataLocation, get_nb_classes

# CONFIGURATION
SPLIT = SplitOptions.SPLIT_ALL

DATA_STUDIED = DataLocation.PLANT
NUM_CLASSES = get_nb_classes(DATA_STUDIED.value + "/train") or get_nb_classes(DATA_STUDIED.value)
NUM_EPOCHS = 50
BATCH_SIZE = 80
TRAIN_SIZE = 0.75
TEST_SIZE = 0.8
CROSS_VALIDATION = False
NB_FOLD = 4 if CROSS_VALIDATION else 1

MODELS = [ModelEnum.CNN5]

SAVE_VALUE = True
OUTPUT_FILE = './resources/data.xlsx'
ROUNDING_DIGIT = 5
