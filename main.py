from learning.trainer import TrainingGenerator
from common.data_imports import SplitOptions
from common.helper import build_model, import_data, ModelEnum, DataLocation, train, get_nb_classes
from common.formatter_from_csv import csv_format_into_folder

# If necessary
FORMAT_FROM_CSV_INTO_FOLDER = False
DATA_STUDIED = DataLocation.PLANT
NUM_CLASSES = get_nb_classes(DATA_STUDIED.value + "/train") | get_nb_classes(DATA_STUDIED.value)
CROSS_VALIDATION = True
NB_FOLD = 4 if CROSS_VALIDATION else 1


def format_csv():
    csv_format_into_folder(location_csv=DataLocation.WHALES.value + "/train.csv",
                           location_folder=DataLocation.WHALES.value + "/train/",
                           location_img=DataLocation.WHALES.value + "/train/train")


def main():
    model = build_model(ModelEnum.CNN2, nb_classes=NUM_CLASSES, depth_input=3)

    data = import_data(batch_size=80, main_folder=DATA_STUDIED, split=SplitOptions.SPLIT_ALL, train_size=0.75,
                       test_size=0.15, k_fold=CROSS_VALIDATION, nb_chunk=NB_FOLD)

    # Saving information for test purposes
    save_value = True
    sheet_name = "P CNN2 CV Test"
    source_to_save_data = './resources/data.xlsx'
    rounding_digit = 5

    for k in range(NB_FOLD):
        sheet_name_fold = sheet_name + f" - fold {k}"
        generator = TrainingGenerator(model=model, data=data.data[k], number_epoch=1, print_intermediate_perf=False,
                                      parameters_data_input=data.parameters_data_input, rounding_digit=rounding_digit,
                                      save_performances=save_value, sheet_name=sheet_name_fold,
                                      location_to_save=source_to_save_data)

        train(generator, k_fold=CROSS_VALIDATION, fold=NB_FOLD)


if __name__ == "__main__":
    # if FORMAT_FROM_CSV_INTO_FOLDER:
    #     format_csv()
    # print(NUM_CLASSES)
    main()
