from enum import Enum
from models.CNN import ModelCnn2Layers, ModelCnn3Layers, ModelCnn4Layers, ModelCnn5Layers, ModelCnn5LayersSmallKernel
from models.Resnet import Resnet, ResnetEndTuned, ResNet34, ResnetLayers3And4Tuned
from common.data_imports import DataImporter, SplitOptions
from common.logger import logger
from learning.trainer import TrainingGenerator
from data_saver.excel_actions import SheetSaver
import os

from models.image_classifiers import InceptionV3

DATA_FOLDER = "data/"


class DataLocation(Enum):
    X_RAY = DATA_FOLDER + "chest-xray-pneumonia/chest_xray/chest_xray"
    HAND = DATA_FOLDER + "leap"
    PLANT = DATA_FOLDER + "plant-seedlings/train"
    WHALES = DATA_FOLDER + "whale"
    BLOOD = DATA_FOLDER + "blood-cells/dataset2-master/dataset2-master/images"


class ModelEnum(Enum):
    CNN2 = ModelCnn2Layers
    CNN3 = ModelCnn3Layers
    CNN4 = ModelCnn4Layers
    CNN5 = ModelCnn5Layers
    CNN5_Small = ModelCnn5LayersSmallKernel
    RESNET = Resnet
    RESNET4TUN = ResnetEndTuned
    RESNET34TUN = ResnetLayers3And4Tuned
    RESNET34 = ResNet34
    INCEPTION3 = InceptionV3


def get_nb_classes(path: str):
    nb_classes = 0
    for _, dirnames, filenames in os.walk(path):
        nb_classes += len(dirnames)
    return nb_classes


def build_model(model: ModelEnum, nb_classes: int, depth_input: int, height_fc: int = 500, dropout_conv: float = None,
                dropout_fc: float = None, batch_norm: bool = False):

    logger.info("Creation of the structure of the models ...")

    if model in (ModelEnum.RESNET, ModelEnum.RESNET34, ModelEnum.RESNET34TUN,
                 ModelEnum.RESNET4TUN, ModelEnum.INCEPTION3):
        return model.value(nb_classes)

    else:
        return model.value(nb_classes, depth_input, height_fc=height_fc, dropout_conv=dropout_conv,
                           dropout_fc=dropout_fc, batch_norm=batch_norm)


def import_data(batch_size: int, main_folder: DataLocation, split: SplitOptions, train_size: float,
                test_size: float = 0.2, k_fold: bool = False, nb_chunk: int = 4):
    logger.info("Creation of the structure of the models ... completed")
    logger.info("Importation and separation of data ... ")

    return DataImporter(batch_size=batch_size, main_folder=main_folder.value, split=split, train_size=train_size,
                        test_size=1 - test_size, k_fold=k_fold, nb_chunk=nb_chunk)


def train(gen: TrainingGenerator, k_fold: bool = False, fold: int = 1):
    logger.info("Importation and separation of data ... completed ")

    logger.info("Training ...." + (f"fold {fold}" if k_fold else ""))
    if gen.save_val:
        # Test if the sheet name is free
        if SheetSaver(gen.sheet_saver.location).test_name_sheet(gen.sheet_name):
            gen.train_model()
            gen.test()
    else:
        gen.train_model()
        gen.test()
