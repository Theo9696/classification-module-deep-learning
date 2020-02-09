import torch.nn as nn
import torch.nn.functional as F
from common.logger import logger

SIZE_IMAGE = (256, 256)


class LayersMaterials:
    def __init__(self, size_in, size_out):
        self._size_in = size_in
        self._size_out = size_out

    def get_size_layer_in(self):
        return self._size_in

    def get_size_layer_out(self):
        return self._size_out

    def apply(self, x):
        return x


class CNNetMaterials(LayersMaterials):
    def __init__(self, num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list[int] = None,
                 dropout: int = -1):
        super(LayersMaterials, self).__init__(num_conv_in, num_conv_out)
        self._stride = stride
        self._kernel_size = kernel_size
        self._pooling = pooling
        self._dropout = dropout
        self._layer = nn.Conv2d(num_conv_in, num_conv_out, kernel_size, stride)
        self._drop = nn.Dropout2d(p=dropout) if dropout > 0 & dropout <= 1 else None

    def get_stride(self):
        return self._stride

    def get_kernel(self):
        return self._kernel_size

    def get_pooling(self):
        return self._pooling

    def get_reduction_size(self):
        return self.get_kernel(), self.get_kernel() if self.get_pooling() is None \
            else self.get_kernel() + self.get_pooling()[0], self.get_kernel() + self.get_pooling()[1]

    def has_dropout(self):
        return self._dropout > 0 & self._dropout <= 1

    def get_dropout(self):
        return self._dropout

    def apply(self, x):

        x = self._layer(x)

        # Dropout
        if self.has_dropout() & (self._drop is not None):
            x = self._drop(x)

        # Non linearity
        x = F.relu(x)

        # Pooling
        pooling = self.get_pooling()
        if pooling:
            x = F.max_pool2d(x, pooling[0], pooling[1])

        return x


class MLPnetMaterials(LayersMaterials):
    def __init__(self, size_in: int, size_out: int, is_last_layer: bool = False):
        super(LayersMaterials).__init__(size_in, size_out)
        self._layer = nn.Linear(size_in, size_out)
        self.is_last_layer = is_last_layer

    def apply(self, x):
        x = self._layer(x)

        if not self.is_last_layer:
            x = F.relu(x)

        return x


class CNN(nn.Module):
    def __init__(self, list_cnn_materials: list[CNNetMaterials], list_mlp_materials: list[MLPnetMaterials],
                 num_classes: int):

        # Check parameters to build a valid model
        self.check_materials(list_cnn_materials, list_mlp_materials, num_classes)

        # Initialisation
        super(CNN, self).__init__()
        self.num_classes = num_classes
        self.cnn_layers = list_cnn_materials
        self.mlp_layers = list_mlp_materials

    def forward(self, x):
        for conv_layer in self.cnn_layers:
            x = conv_layer.apply(x)

        x = x.view(-1, 4 * 4 * self.cnn_layers[-1].get_size_layer_out())  # MLP => prend un vecteur depuis la matrice

        for fc_layer in self.mlp_layers:
            x = fc_layer.apply(x)
        return x

    @staticmethod
    def check_materials(list_cnn_materials: list[CNNetMaterials], list_mlp_materials: list[MLPnetMaterials],
                        num_classes: int):

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

        image_after_cnn_layers = (image_after_layer[0] / list_cnn_materials[-1].get_reduction_size()[0],
                                  image_after_layer[1] / list_cnn_materials[-1].get_reduction_size()[1])

        assert list_mlp_materials[0].get_size_layer_in() == image_after_cnn_layers[0] * image_after_cnn_layers[1] * \
               list_cnn_materials[-1].get_size_layer_out(), logger.error("Problem of size connections cnn to mlp")

        assert list_mlp_materials[-1].get_size_layer_out() == num_classes, logger.error(
            "Size of ending layer must be the number of classes")


