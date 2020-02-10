import torch.nn as nn
from common.logger import logger
from common.layers_builders import CNNetMaterials, MLPnetMaterials

SIZE_IMAGE = (256, 256)

FINAL_SIZE = 61


class CNN(nn.Module):
    def __init__(self, list_cnn_materials: list, list_mlp_materials: list, num_classes: int):
        super(CNN, self).__init__()

        # Check parameters to build a valid model
        self.check_materials(list_cnn_materials, list_mlp_materials, num_classes)

        # Initialisation
        self.num_classes = num_classes

        # Generate parameters
        self.layers = self.generateModuleList(list_cnn_materials + list_mlp_materials)
        self.cnn_layers = list_cnn_materials
        self.mlp_layers = list_mlp_materials

    def forward(self, x):
        for conv_layer in self.cnn_layers:
            x = conv_layer.apply(x)

        x = x.view(-1, FINAL_SIZE * FINAL_SIZE * self.cnn_layers[-1].get_size_layer_out())  # MLP => prend un vecteur depuis la matrice

        for fc_layer in self.mlp_layers:
            x = fc_layer.apply(x)
        return x

    @staticmethod
    def generateModuleList(list_layers: list):
        modules = []
        for e in list_layers:
            modules.append(e.layer)
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
