from enum import Enum
from models.CNN import ModelCnn2Layers, ModelCnn3Layers, ModelCnn4Layers
from models.Resnet import Resnet
from common.data_imports import DataImporter, SplitOptions
from common.logger import logger
from learning.trainer import TrainingGenerator
from data_saver.excel_actions import SheetSaver
import os


DATA_FOLDER = "data/"

class DataLocation(Enum):
    X_RAY = DATA_FOLDER + "chest-xray-pneumonia/chest_xray/chest_xray"
    HAND = DATA_FOLDER + "leap"
    PLANT = DATA_FOLDER + "plant-seedlings"
    WHALES = DATA_FOLDER + "whale"


class ModelEnum(Enum):
    CNN2 = ModelCnn2Layers
    CNN3 = ModelCnn3Layers
    CNN4 = ModelCnn4Layers
    RESNET = Resnet


def get_nb_classes(path: str):
    nb_classes = 0
    for _, dirnames, filenames in os.walk(path):
        nb_classes += len(dirnames)
    return nb_classes


def build_model(model: ModelEnum, nb_classes: int, depth_input: int):
    logger.info("Creation of the structure of the models ...")

    if model is ModelEnum.RESNET:
        return model.value(nb_classes)

    else:
        return model.value(nb_classes, depth_input)


def import_data(batch_size: int, main_folder: DataLocation, split: SplitOptions, train_size: float,
                test_size: float = 0.2):
    logger.info("Creation of the structure of the models ... completed")
    logger.info("Importation and separation of data ... ")

    return DataImporter(batch_size=batch_size, main_folder=main_folder.value, split=split, train_size=train_size,
                        test_size=1 - test_size)


def train(gen: TrainingGenerator):
    logger.info("Importation and separation of data ... completed ")

    logger.info("Training ....")
    if gen.save_val:
        # Test if the sheet name is free
        if SheetSaver(gen.sheet_saver.location).test_name_sheet(gen.sheet_name):
            gen.train()
            gen.test()
    else:
        gen.train()
        gen.test()
