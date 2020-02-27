import torch.nn as nn

from common.layers_builders import CNNetMaterials, MLPnetMaterials
from common.logger import logger
from enum import Enum
from models.Model import Model

SIZE_IMAGE = (256, 256)


class CNNParameters(Enum):
    NB_LAYERS_CNN = "nb cnn layers"
    POOLING = "pooling"
    STRIDE = "stride"
    KERNEL = "kernel size"
    HEIGHT_FC_CNN = "height MLP layer for CNN"


class CNN(nn.Module):
    def __init__(self, list_cnn_materials: list, list_mlp_materials: list, num_classes: int, final_size: int):
        super(CNN, self).__init__()

        # Check parameters to build a valid model
        self.check_materials(list_cnn_materials, list_mlp_materials, num_classes)

        # Initialisation
        self.num_classes = num_classes

        # Generate parameters
        self.layers = self.generate_module_list(list_cnn_materials + list_mlp_materials)
        self.cnn_layers = list_cnn_materials
        self.mlp_layers = list_mlp_materials
        self.final_size = final_size

    def forward(self, x):
        for conv_layer in self.cnn_layers:
            x = conv_layer.apply(x)

        x = x.view(-1, self.final_size * self.final_size * self.cnn_layers[
            -1].get_size_layer_out())  # MLP => prend un vecteur depuis la matrice

        for fc_layer in self.mlp_layers:
            x = fc_layer.apply(x)
        return x

    @staticmethod
    def generate_module_list(list_layers: list):
        modules = [e.layer for e in list_layers]
        return nn.ModuleList(modules)

    @staticmethod
    def check_materials(list_cnn_materials: list, list_mlp_materials: list, num_classes: int):

        # Check basic arguments shapes
        assert isinstance(list_cnn_materials, list), logger.error("Wrong type of argument to initialize the model")
        assert isinstance(list_mlp_materials, list), logger.error("Wrong type of argument to initialize the model")
        assert len(list_cnn_materials) > 0, logger.error("The model needs at least one CNN layer")
        assert len(list_mlp_materials) > 0, logger.error("The model needs at least one fully connected layer")
        assert num_classes > 0, logger.error(
            f" the model needs at least one class, but found : {num_classes}")

        # Check sizes of layers
        image_after_layer = SIZE_IMAGE
        for k in range(len(list_cnn_materials) - 1):
            image_after_layer = (image_after_layer[0] / list_cnn_materials[k].get_reduction_size()[0],
                                 image_after_layer[1] / list_cnn_materials[k].get_reduction_size()[1])

            assert list_cnn_materials[k].get_size_layer_out() == list_cnn_materials[k + 1].get_size_layer_in(), \
                logger.error(
                    f"Problem in convolutional layer sizes, out: {list_cnn_materials[k].get_size_layer_out()}"
                    f", in: {list_cnn_materials[k + 1].get_size_layer_in()}")

            assert list_cnn_materials[k + 1].get_size_layer_in() > 0, logger.error(
                "the layers must have a positive height")

        # TODO check correctly reduction pooling / and CNN -
        # image_after_cnn_layers = (image_after_layer[0] / list_cnn_materials[-1].get_reduction_size()[0],
        #                           image_after_layer[1] / list_cnn_materials[-1].get_reduction_size()[1])
        #
        # assert list_mlp_materials[0].get_size_layer_in() == image_after_cnn_layers[0] * image_after_cnn_layers[1] * \
        #        list_cnn_materials[-1].get_size_layer_out(), logger.error("Problem of size connections cnn to mlp")
        #
        assert list_mlp_materials[-1].get_size_layer_out() == num_classes, logger.error(
            "Size of ending layer must be the number of classes")


class ModelCNN(Model):
    def __init__(self, nb_classes: int, depth_input: int = 1):
        super().__init__(nb_classes_out=nb_classes)
        self.nb_layers = 0
        self.pooling = 0
        self.stride = 0
        self.kernel = 0
        self.depth_input = depth_input
        self.height_fc = 0

    def get_parameters(self):
        return {
            CNNParameters.NB_LAYERS_CNN.value: self.nb_layers,
            CNNParameters.POOLING.value: self.pooling,
            CNNParameters.STRIDE.value: self.stride,
            CNNParameters.KERNEL.value: self.kernel if type(self.kernel) is int else str(self.kernel),
            CNNParameters.HEIGHT_FC_CNN.value: self.height_fc
        }


class ModelCnn2Layers(ModelCNN):
    def __init__(self, nb_classes: int, depth_input: int = 1, dropout: bool = False, height_fc: int = 500,
                 batch_norm: bool = False):
        super().__init__(nb_classes, depth_input)
        self.nb_layers = 2
        self.num_conv_1 = 10  # try 32
        self.num_conv_2 = 20  # try 64
        self.height_fc = height_fc  # try 1024
        self.pooling = 2
        self.stride = 1
        self.kernel = 5
        self.final_size = 61
        # num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,
        layer_1 = CNNetMaterials(depth_input, self.num_conv_1, self.stride, self.kernel,
                                 [self.pooling, self.pooling], dropout=dropout, batch_norm=batch_norm)
        layer_2 = CNNetMaterials(self.num_conv_1, self.num_conv_2, self.stride, self.kernel,
                                 [self.pooling, self.pooling], dropout=dropout, batch_norm=batch_norm)
        layer_3 = MLPnetMaterials(self.num_conv_2 * self.final_size * self.final_size, self.height_fc)
        layer_4 = MLPnetMaterials(self.height_fc, self.nb_classes_out, is_last_layer=True)
        self.model = CNN([layer_1, layer_2], [layer_3, layer_4], self.nb_classes_out, self.final_size)


class ModelCnn3Layers(ModelCNN):
    def __init__(self, nb_classes: int, depth_input: int = 1, dropout: bool = False, height_fc: int = 500,
                 batch_norm: bool = False):
        super().__init__(nb_classes, depth_input)
        self.nb_layers = 3
        num_conv_1 = 10
        num_conv_2 = 20
        num_conv_3 = 35
        self.height_fc = height_fc
        final_size = 28

        self.pooling = 2
        self.stride = 1
        self.kernel = [5, 5, 6]

        # num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,
        layer_1 = CNNetMaterials(self.depth_input, num_conv_1, self.stride, self.kernel[0],
                                 [self.pooling, self.pooling], dropout=dropout, batch_norm=batch_norm)
        layer_2 = CNNetMaterials(num_conv_1, num_conv_2, self.stride, self.kernel[1], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_3 = CNNetMaterials(num_conv_2, num_conv_3, self.stride, self.kernel[2], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_4 = MLPnetMaterials(num_conv_3 * final_size * final_size, self.height_fc)
        layer_5 = MLPnetMaterials(self.height_fc, self.nb_classes_out, is_last_layer=True)

        self.model = CNN([layer_1, layer_2, layer_3], [layer_4, layer_5], self.nb_classes_out, final_size)


class ModelCnn4Layers(ModelCNN):
    def __init__(self, nb_classes: int, depth_input: int = 1, dropout: bool = False, height_fc: int = 500,
                 batch_norm: bool = False):
        super().__init__(nb_classes, depth_input)
        self.nb_layers = 4
        num_conv_1 = 10
        num_conv_2 = 20
        num_conv_3 = 36
        num_conv_4 = 50
        self.height_fc = height_fc

        self.pooling = 2
        self.stride = 1
        self.kernel = [5, 5, 6, 5]

        final_size = 12

        # num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,

        layer_1 = CNNetMaterials(depth_input, num_conv_1, self.stride, self.kernel[0], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_2 = CNNetMaterials(num_conv_1, num_conv_2, self.stride, self.kernel[1], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_3 = CNNetMaterials(num_conv_2, num_conv_3, self.stride, self.kernel[2], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_4 = CNNetMaterials(num_conv_3, num_conv_4, self.stride, self.kernel[3], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_5 = MLPnetMaterials(num_conv_4 * final_size * final_size, self.height_fc)
        layer_6 = MLPnetMaterials(self.height_fc, nb_classes, is_last_layer=True)

        self.model = CNN([layer_1, layer_2, layer_3, layer_4], [layer_5, layer_6], nb_classes, final_size)


class ModelCnn5Layers(ModelCNN):
    def __init__(self, nb_classes: int, depth_input: int = 1, dropout: bool = False, height_fc: int = 500,
                 batch_norm: bool = False):
        super().__init__(nb_classes, depth_input)
        self.nb_layers = 5
        num_conv_1 = 10
        num_conv_2 = 20
        num_conv_3 = 36
        num_conv_4 = 50
        num_conv_5 = 75
        self.height_fc = height_fc

        self.pooling = 2
        self.stride = 1
        self.kernel = [5, 5, 6, 5, 5]

        final_size = 4

        # num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,

        layer_1 = CNNetMaterials(depth_input, num_conv_1, self.stride, self.kernel[0], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_2 = CNNetMaterials(num_conv_1, num_conv_2, self.stride, self.kernel[1], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_3 = CNNetMaterials(num_conv_2, num_conv_3, self.stride, self.kernel[2], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_4 = CNNetMaterials(num_conv_3, num_conv_4, self.stride, self.kernel[3], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_5 = CNNetMaterials(num_conv_4, num_conv_5, self.stride, self.kernel[4], [self.pooling, self.pooling],
                                 dropout=dropout, batch_norm=batch_norm)
        layer_6 = MLPnetMaterials(num_conv_5 * final_size * final_size, self.height_fc)
        layer_7 = MLPnetMaterials(self.height_fc, nb_classes, is_last_layer=True)

        self.model = CNN([layer_1, layer_2, layer_3, layer_4, layer_5], [layer_6, layer_7], nb_classes, final_size)
