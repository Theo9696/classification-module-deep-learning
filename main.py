import time

from common.logger import logger
from learning.trainer import TrainingGenerator
from common.data_imports import SplitOptions
from common.helper import build_model, import_data, ModelEnum, DataLocation, train, get_nb_classes
from common.formatter_from_csv import csv_format_into_folder

# CONFIGURATION
FORMAT_FROM_CSV_INTO_FOLDER = False  # If necessary
SPLIT = SplitOptions.SPLIT_ALL

DATA_STUDIED = DataLocation.PLANT
NUM_CLASSES = get_nb_classes(DATA_STUDIED.value + "/train") or get_nb_classes(DATA_STUDIED.value)
NUM_EPOCHS = 10
BATCH_SIZE = 40
TRAIN_SIZE = 0.75
TEST_SIZE = 1 - TRAIN_SIZE
CROSS_VALIDATION = True
NB_FOLD = 4 if CROSS_VALIDATION else 1

MODELS = [ModelEnum.RESNET]

SAVE_VALUE = True
OUTPUT_FILE = './resources/data.xlsx'
ROUNDING_DIGIT = 5


def format_csv():
    csv_format_into_folder(location_csv=DataLocation.WHALES.value + "/train.csv",
                           location_folder=DataLocation.WHALES.value + "/train/",
                           location_img=DataLocation.WHALES.value + "/train/train")


def main():
    data = import_data(batch_size=BATCH_SIZE, main_folder=DATA_STUDIED, split=SPLIT, train_size=TRAIN_SIZE,
                       test_size=TEST_SIZE, k_fold=CROSS_VALIDATION, nb_chunk=NB_FOLD)

    for model_type in MODELS:
        sheet_name = f"{DATA_STUDIED.name} {model_type.name} - {NUM_EPOCHS} epoch"

        for k in range(NB_FOLD):
            sheet_name_fold = sheet_name + f" - fold {k}"
            model = build_model(model_type, nb_classes=NUM_CLASSES, depth_input=3)
            generator = TrainingGenerator(model=model, data=data.data[k],
                                          number_epoch=NUM_EPOCHS,
                                          print_intermediate_perf=False,
                                          parameters_data_input=data.parameters_data_input,
                                          rounding_digit=ROUNDING_DIGIT,
                                          save_performances=SAVE_VALUE,
                                          sheet_name=sheet_name_fold,
                                          location_to_save=OUTPUT_FILE)

            train(generator, k_fold=CROSS_VALIDATION, fold=k)


if __name__ == "__main__":
    # if FORMAT_FROM_CSV_INTO_FOLDER:
    #     format_csv()
    # print(NUM_CLASSES)
    t0 = time.time()
    main()
    print(f"Process terminated in {time.time() - t0:.2f} s")
