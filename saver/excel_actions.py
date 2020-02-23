import xlrd
from xlutils.copy import copy
from enum import Enum


class SheetNames(Enum):
    PARAMETERS = "parameters"
    LOSS_FUNCTION = "loss function value"
    TRAIN_ERROR = "train error"
    VAL_ERROR = "val error"
    TEST_ERROR = "test error"


class ParametersNames(Enum):
    MODEL = "model"
    MOMENTUM = "momentum"
    LEARNING_RATE = "learning rate"
    DROPOUT = "dropout"
    NB_EPOCH = "nb epoch"


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

    def write_dic(self, dic: dict, sheet_name: str):
        print(dic)
        self.write_list(self.format_dic(dic), sheet_name)

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
    def format_parameters(parameters: dict):
        list_to_save = []
        if (parameters is not None) & (type(parameters) is dict):
            index = 0
            for param in parameters:
                list_to_save.append([0, index + 1, param])
                list_to_save.append([1, index + 1, parameters[param]])
                index += 1

        # The lign of index 3 will be filled with Epoch 0 Epoch 1 Epoch 2 ...
        index_name = 3
        nb_epoch = parameters[str(ParametersNames.NB_EPOCH.value)]
        print(nb_epoch)
        if nb_epoch is not None:
            for k in range(nb_epoch):
                list_to_save.append([index_name, 2 + k, f"epoch {k}"])

        return list_to_save

    @staticmethod
    def format_dic(dic: dict):
        parameters = dic[SheetNames.PARAMETERS.value]
        list_to_save = SheetSaver.format_parameters(parameters)

        index_name = 3
        for name_element in dic:
            if name_element is not SheetNames.PARAMETERS.value:
                index_name += 1
                list_to_save.append([index_name, 1, name_element])
                for element in dic[name_element]:
                    # element looks like (epoch, value)
                    list_to_save.append([index_name, 1 + element[0], element[1]])
        return list_to_save


if __name__ == "__main__":
    dic = {SheetNames.PARAMETERS.value: {ParametersNames.LEARNING_RATE.value: 5, ParametersNames.MOMENTUM.value: 2,
                                         ParametersNames.NB_EPOCH.value: 3},
           SheetNames.TRAIN_ERROR.value: [(1, 0.3), (2, 0.35), (3, 0.41)],
           SheetNames.LOSS_FUNCTION.value: [(1, 5), (2, 2), (3, 0.27)],
           SheetNames.TEST_ERROR.value: [(3, 0.41)]}
    saver = SheetSaver("../resources/data.xlsx")
    print(saver.write_dic(dic=dic, sheet_name="essai_11"))
    saver.close_file()
