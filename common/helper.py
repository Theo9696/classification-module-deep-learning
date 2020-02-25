from enum import Enum
from models.CNN import ModelCnn2Layers, ModelCnn3Layers, ModelCnn4Layers
from models.Resnet import Resnet
from common.data_imports import DataImporter, SplitOptions
from common.logger import logger
from learning.trainer import TrainingGenerator
from data_saver.excel_actions import SheetSaver


class DataLocation(Enum):
    X_RAY = "chest-xray-pneumonia/chest_xray/chest_xray"
    HAND = "leap"


class ModelEnum(Enum):
    CNN2 = ModelCnn2Layers
    CNN3 = ModelCnn3Layers
    CNN4 = ModelCnn4Layers
    RESNET = Resnet


def build_model(model: ModelEnum, nb_classes: int, depth_input: int):
    logger.info("Creation of the structure of the models ...")

    if model is ModelEnum.RESNET:
        return model.value(nb_classes)

    else:
        return model.value(nb_classes, depth_input)


def import_data(batch_size: int, main_folder: DataLocation, split: SplitOptions, train_size: float):
    logger.info("Creation of the structure of the models ... completed")
    logger.info("Importation and separation of data ... ")

    return DataImporter(batch_size=batch_size, main_folder=main_folder.value, split=split, train_size=train_size)


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
