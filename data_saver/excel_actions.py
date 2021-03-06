import xlrd
from xlutils.copy import copy
from enum import Enum


class SheetNames(Enum):
    PARAMETERS = "parameters"
    PARAMETERS_MODELS = "model parameters"
    TRAINING = "training"
    RESULT = "Result"


class ParametersNames(Enum):
    MODEL = "model"
    MOMENTUM = "momentum"
    LEARNING_RATE = "learning rate"
    DROPOUT = "dropout"
    NB_EPOCH = "nb epoch"
    TIME = "time"
    NB_TRAIN = "nb samples train"
    NB_VAL = "nb samples val"
    NB_TEST = "nb samples test"
    SIZE_IMAGE_INPUT_MODEL = "resized size of images"


class Result(Enum):
    TP = "True Positive Test"
    TN = "True Negative Test"
    FP = "False Positive Test"
    FN = "False Negative Test"
    LOSS = "Average loss Test"
    ACCURACY = "Accuracy Test"
    PRECISION = "Precision Test"
    RECALL = "Recall Test"
    CONFUSION_MATRIX = "Confusion matrix Test"
    PRETTY_CONFUSION = "pretty confusion matrix"


class TrainingResult(Enum):
    TP = "True Positive Val"
    TN = "True Negative Val"
    FP = "False Positive Val"
    FN = "False Negative Val"
    CONFUSION_MATRIX = "Confusion matrix"
    LOSS_TRAIN = "Average loss Train"
    LOSS_VAL = "Average loss Val"
    ACCURACY = "Accuracy Val"
    PRECISION = "Precision Val"
    RECALL = "Recall Val"


class SheetSaver:
    def __init__(self, location: str):
        self.location = location
        self.workbook = xlrd.open_workbook(location)

    def read(self, sheet_index: int, cell: tuple):
        sheet = self.workbook.sheet_by_index(sheet_index)
        return sheet.cell_value(cell[0], cell[1])

    def write_list(self, list_element: list, sheet_name: str):
        w_write = copy(self.workbook)
        w_sheet = None

        try:
            w_write.add_sheet(sheet_name)
            w_sheet = w_write.get_sheet(sheet_name)
        except IndexError as e:
            print(f"The sheet has no sheet of index : {sheet_name} \n Original error was: {e}")
        except Exception as e:
            print(f"[Error during saving process] Original error was: {e}")

        if w_sheet is not None:
            for element in list_element:
                w_sheet.write(element[0], element[1], element[2])
            w_write.save(self.location)

    def write_dic(self, dictionary: dict, sheet_name: str):
        # print(dictionary)
        self.write_list(self.format_dic(dictionary), sheet_name)

    def close_file(self):
        self.workbook.release_resources()

    def test_name_sheet(self, sheet_name: str):
        w_write = copy(self.workbook)

        try:
            w_write.add_sheet(sheet_name)
        except Exception as e:
            print(f"This name of sheet is already used, please change it \n Original error: {e}")
            return False
        return True

    @staticmethod
    def format_parameters(parameters: dict, index_lign: int, list_to_save: list):
        if (parameters is not None) & (type(parameters) is dict):
            index = 0
            for param in parameters:
                list_to_save.append([index_lign, index + 1, param])
                list_to_save.append([index_lign + 1, index + 1, parameters[param]])
                index += 1

        return list_to_save

    @staticmethod
    def initialize_title_epoch(nb_epoch: int, list_to_save: list):
        # The lign of index 3 will be filled with Epoch 0 Epoch 1 Epoch 2 ...
        index_name = 8
        if nb_epoch is not None:
            for k in range(nb_epoch):
                list_to_save.append([index_name, 2 + k, f"epoch {k}"])

    @staticmethod
    def format_dic(dic: dict):
        parameters = dic[SheetNames.PARAMETERS.value]
        list_to_save = SheetSaver.format_parameters(parameters, 0, [])
        SheetSaver.format_parameters(dic[SheetNames.PARAMETERS_MODELS.value], 2, list_to_save)
        SheetSaver.format_parameters(dic[SheetNames.RESULT.value], 4, list_to_save)
        SheetSaver.initialize_title_epoch(parameters[ParametersNames.NB_EPOCH.value], list_to_save)

        index_name = 8
        training_values = dic[SheetNames.TRAINING.value]
        for name_element in training_values:
            index_name += 1
            list_to_save.append([index_name, 1, name_element])
            for element in training_values[name_element]:
                # element looks like (epoch, value)
                list_to_save.append([index_name, 1 + element[0], element[1]])
        return list_to_save


if __name__ == "__main__":
    dic = {SheetNames.PARAMETERS.value: {ParametersNames.LEARNING_RATE.value: 5, ParametersNames.MOMENTUM.value: 2,
                                         ParametersNames.NB_EPOCH.value: 3},
           SheetNames.PARAMETERS_MODELS.value: {"nb layers": 3, "kernel": 4},
           SheetNames.TRAINING.value: {
               TrainingResult.LOSS_VAL.value: [(1, 5), (2, 2), (3, 0.27)],
               TrainingResult.LOSS_TRAIN.value: [(1, 3), (2, 1), (3, 0.17)],
               TrainingResult.ACCURACY.value: [(1, 0.3), (2, 0.35), (3, 0.41)],
           },
           SheetNames.RESULT.value: {
               Result.FP.value: 5,
               Result.TN.value: 100,
               Result.TP.value: 150,
               Result.FN.value: 50
           }}
    saver = SheetSaver("../resources/data.xlsx")
    print(saver.write_dic(dictionary=dic, sheet_name="essai_model 4"))
    saver.close_file()
