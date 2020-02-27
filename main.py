from learning.trainer import TrainingGenerator
from common.data_imports import SplitOptions
from common.helper import build_model, import_data, ModelEnum, DataLocation, train, get_nb_classes
from common.formatter_from_csv import csv_format_into_folder

# If necessary
FORMAT_FROM_CSV_INTO_FOLDER = False
DATA_STUDIED = DataLocation.PLANT
NUM_CLASSES = get_nb_classes(DATA_STUDIED.value + "/train") | get_nb_classes(DATA_STUDIED.value)


def format_csv():
    csv_format_into_folder(location_csv=DataLocation.WHALES.value + "/train.csv",
                           location_folder=DataLocation.WHALES.value + "/train/",
                           location_img=DataLocation.WHALES.value + "/train/train")


def main():
    model = build_model(ModelEnum.CNN4, nb_classes=NUM_CLASSES, depth_input=3)

    data = import_data(batch_size=40, main_folder=DATA_STUDIED, split=SplitOptions.SPLIT_ALL, train_size=0.75,
                       test_size=0.15)

    # Saving information for test purposes
    save_value = True
    sheet_name = "Plant CNN4 - 15 epoch"
    source_to_save_data = './resources/data.xlsx'
    rounding_digit = 5

    generator = TrainingGenerator(model=model, data=data, number_epoch=15, print_intermediate_perf=False,
                                  parameters_data_input=data.parameters_data_input, rounding_digit=rounding_digit,
                                  save_performances=save_value, sheet_name=sheet_name,
                                  location_to_save=source_to_save_data)

    train(generator)


if __name__ == "__main__":
    # if FORMAT_FROM_CSV_INTO_FOLDER:
    #     format_csv()
    # print(NUM_CLASSES)
    main()
