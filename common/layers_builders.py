import torch.nn as nn
import torch.nn.functional as F


class LayersMaterials:
    def __init__(self, size_in: int, size_out: int):
        self._size_in = size_in
        self._size_out = size_out

    def get_size_layer_in(self):
        return self._size_in

    def get_size_layer_out(self):
        return self._size_out

    def apply(self, x):
        return x


class CNNetMaterials(LayersMaterials):
    """
    A class enabling to build and use a CNN layer
    arg: height_in, height_out, stride, kernel size, pooling [x,y], dropout
    """

    def __init__(self, num_conv_in: int, num_conv_out: int, stride: int, kernel_size: int, pooling: list,
                 dropout: float = None, batch_norm: bool = False):
        super().__init__(num_conv_in, num_conv_out)
        self._stride = stride
        self._kernel_size = kernel_size
        self._pooling = pooling
        self._dropout = dropout
        self._batch_norm = batch_norm
        if batch_norm:
            self.layer = nn.Sequential(nn.Conv2d(num_conv_in, num_conv_out, kernel_size, stride),
                                       nn.BatchNorm2d(num_conv_out))
        else:
            self.layer = nn.Conv2d(num_conv_in, num_conv_out, kernel_size, stride)
        self.drop = nn.Dropout2d(p=dropout) if (dropout is not None) else None

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
        return self._dropout is not None

    def get_dropout(self):
        return self._dropout

    def apply(self, x):

        x = self.layer(x)

        # # Batch norm
        # if self._batch_norm:
        #     x = self.batch_norm(x)

        # Non linearity
        x = F.relu(x)

        # Pooling
        pooling = self.get_pooling()
        if pooling:
            x = F.max_pool2d(x, pooling[0], pooling[1])

        # Dropout
        if self.has_dropout() & (self.drop is not None):
            x = self.drop(x)

        return x


class MLPnetMaterials(LayersMaterials):
    def __init__(self, size_in: int, size_out: int, is_last_layer: bool = False, dropout: float = None):
        super().__init__(size_in, size_out)
        self.layer = nn.Linear(size_in, size_out)
        self.drop = nn.Dropout2d(p=dropout) if (dropout is not None) else None
        self.is_last_layer = is_last_layer if is_last_layer is not None else False
        self.dropout = dropout

    def apply(self, x):
        if (self.dropout is not None) & (not self.is_last_layer):
            x = self.drop(x)

        x = self.layer(x)

        if not self.is_last_layer:
            x = F.relu(x)

        return x
