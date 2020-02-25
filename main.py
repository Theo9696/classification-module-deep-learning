from learning.trainer import TrainingGenerator
from common.data_imports import SplitOptions
from common.helper import build_model, import_data, ModelEnum, DataLocation, train

NUM_CLASSES = 2

model = build_model(ModelEnum.CNN2, nb_classes=NUM_CLASSES, depth_input=3)

data = import_data(batch_size=20, main_folder=DataLocation.X_RAY, split=SplitOptions.SPLIT_TRAIN, train_size=0.75)

# Saving information for test purposes
SAVE_VALUE = True
SHEET_NAME = "CNN 2 - Test 3"
SOURCE_TO_SAVE_DATA = './resources/data.xlsx'

generator = TrainingGenerator(model=model, data=data, number_epoch=1, print_intermediate_perf=False,
                              parameters_data_input=data.parameters_data_input,
                              save_performances=SAVE_VALUE, sheet_name=SHEET_NAME, location_to_save=SOURCE_TO_SAVE_DATA)

train(generator)
