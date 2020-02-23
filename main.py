from learning.trainer import TrainingGenerator
from models.Resnet import Resnet
from common.data_imports import DataImporter
from common.logger import logger
from data_saver.excel_actions import SheetSaver
from models.CNN import ModelCnn2Layers, ModelCnn3Layers, ModelCnn4Layers

logger.info("Creation of the structure of the models ...")

NUM_CLASSES = 2
model = ModelCnn2Layers(nb_classes=NUM_CLASSES, depth_input=3)
# model = ModelCnn3Layers(nb_classes=NUM_CLASSES, depth_input=3)
# model = ModelCnn4Layers(nb_classes=NUM_CLASSES, depth_input=3)
# model = Resnet(NUM_CLASSES)

logger.info("Creation of the structure of the models ... completed")
logger.info("Importation and separation of data ... ")

# Data imports and format
xray = "chest-xray-pneumonia/chest_xray/chest_xray"
hand = "leap"
data = DataImporter(batch_size=20, main_folder=xray, split=False)
logger.info("Importation and separation of data ... completed ")

# Saving information for test purposes
SAVE_VALUE = True
SHEET_NAME = "test time"
SOURCE_TO_SAVE_DATA = './resources/data.xlsx'

logger.info("Training ....")

gen = TrainingGenerator(model=model, data=data, number_epoch=1, print_val=False,
                        save_val=SAVE_VALUE, sheet_name=SHEET_NAME, location_to_save=SOURCE_TO_SAVE_DATA)

if SAVE_VALUE:
    # Test if the sheet name is free
    if SheetSaver(SOURCE_TO_SAVE_DATA).test_name_sheet(SHEET_NAME):
        gen.train()
        gen.test()
else:
    gen.train()
    gen.test()
