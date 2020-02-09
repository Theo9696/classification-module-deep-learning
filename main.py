from learning.trainer import TrainingGenerator
from models.CNN import CNN, CNNetMaterials, MLPnetMaterials
from common.data_imports import dataImporter
from common.logger import logger


logger.info("Creation of the structure of the models ...")

NUM_CLASSES = 10
NUM_CONV_1 = 10  # try 32
NUM_CONV_2 = 20  # try 64
NUM_FC = 500  # try 1024


# num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,

layer_1 = CNNetMaterials(1, NUM_CONV_1, 1, 5, [2,2])
layer_2 = CNNetMaterials(NUM_CONV_1, NUM_CONV_2, 1, 5, [2,2])
layer_3 = MLPnetMaterials(NUM_CONV_2*4*4, NUM_FC)
layer_4 = MLPnetMaterials(NUM_FC, NUM_CLASSES, is_last_layer=True)

model = CNN([layer_1, layer_2], [layer_3, layer_4], NUM_CLASSES)

logger.info("Creation of the structure of the models ... completed")
logger.info("Importation and separation of data ... ")


data = dataImporter('./data', './data', 100)
logger.info("Importation and separation of data completed ")


logger.info("Training ....")

gen = TrainingGenerator(model, data)
gen.train()
logger.info("Training completed")
