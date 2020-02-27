import time

from learning.trainer import TrainingGenerator
from common.helper import build_model, import_data, train
from config import *


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
    # print(NUM_CLASSES)
    t0 = time.time()
    main()
    print(f"Process terminated in {time.time() - t0:.2f} s")
