from common.data_imports import SplitOptions
from common.helper import ModelEnum, DataLocation, get_nb_classes, build_model


class ModelCreator:
    def __init__(self, model_type: ModelEnum, nb_classes: int, depth_input: int, batch_norm: bool = False,
                 dropout_conv: float = None, dropout_fc: float = None, height_fc: int = None):
        self.model = build_model(model_type, nb_classes=nb_classes, depth_input=depth_input, batch_norm=batch_norm,
                                 dropout_conv=dropout_conv, dropout_fc=dropout_fc, height_fc=height_fc)
        self.name = model_type.name


# CONFIGURATION
SPLIT = SplitOptions.SPLIT_ALL

DATA_STUDIED = DataLocation.PLANT
NUM_CLASSES = get_nb_classes(DATA_STUDIED.value + "/train") or get_nb_classes(DATA_STUDIED.value)
NUM_EPOCHS = 39
BATCH_SIZE = 60
TRAIN_SIZE = 0.75
TEST_SIZE = 0.80
CROSS_VALIDATION = False
NB_FOLD = 4 if CROSS_VALIDATION else 1
LEARNING_RATE = 0.0005

MODELS_TYPE = [ModelEnum.CNN5]

MODELS = []

for model_type in MODELS_TYPE:
    MODELS.append(ModelCreator(model_type=model_type, nb_classes=NUM_CLASSES, depth_input=3,
                               batch_norm=True, height_fc=500, dropout_conv=0.05))
    MODELS.append(ModelCreator(model_type=model_type, nb_classes=NUM_CLASSES, depth_input=3,
                               batch_norm=True, height_fc=500, dropout_conv=0.1))
    MODELS.append(ModelCreator(model_type=model_type, nb_classes=NUM_CLASSES, depth_input=3,
                               batch_norm=True, height_fc=500))

SAVE_VALUE = True
OUTPUT_FILE = './resources/data.xlsx'
ROUNDING_DIGIT = 5
