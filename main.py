from learning.trainer import TrainingGenerator
from models.CNN import CNN
from models.Resnet import Resnet
from common.layers_builders import CNNetMaterials, MLPnetMaterials
from common.data_imports import DataImporter
from common.logger import logger
from saver.excel_actions import SheetSaver

logger.info("Creation of the structure of the models ...")

NUM_CLASSES = 2
NUM_CONV_1 = 10  # try 32
NUM_CONV_2 = 20  # try 64
NUM_FC = 500  # try 1024

FINAL_SIZE = 61


# num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,

layer_1 = CNNetMaterials(3, NUM_CONV_1, 1, 5, [2, 2])
layer_2 = CNNetMaterials(NUM_CONV_1, NUM_CONV_2, 1, 5, [2, 2])
layer_3 = MLPnetMaterials(NUM_CONV_2 * FINAL_SIZE * FINAL_SIZE, NUM_FC)
layer_4 = MLPnetMaterials(NUM_FC, NUM_CLASSES, is_last_layer=True)

cnn = CNN([layer_1, layer_2], [layer_3, layer_4], NUM_CLASSES)
resnet = Resnet(NUM_CLASSES).model

logger.info("Creation of the structure of the models ... completed")
logger.info("Importation and separation of data ... ")

xray = "chest-xray-pneumonia/chest_xray/chest_xray"
hand = "leap"
data = DataImporter(batch_size=20, main_folder=xray, split=False)
logger.info("Importation and separation of data ... completed ")

logger.info("Training ....")

SAVE_VALUE = True
SHEET_NAME = "Essai Complet"
SOURCE_TO_SAVE_DATA = './resources/data.xlsx'

gen = TrainingGenerator(model=cnn, data=data, number_epoch=1, print_val=False,
                        save_val=SAVE_VALUE, sheet_name=SHEET_NAME, location_to_save=SOURCE_TO_SAVE_DATA)

if SAVE_VALUE:
    # Test if the sheet name is free
    if SheetSaver(SOURCE_TO_SAVE_DATA).test_name_sheet(SHEET_NAME):
        gen.train()
        gen.test()
else:
    gen.train()
    gen.test()
