# -*- coding: utf-8 -*-
from torch import nn
from torchvision import models

from models.Model import Model


class AlexNet(Model):  # todo: bug ?
    def __init__(self, nb_classes):
        super().__init__(nb_classes_out=nb_classes)
        self.model = models.alexnet(pretrained=True)
        # self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.nb_classes_out, bias=True)


class InceptionV3(Model):
    # TODO: bad kernel size by default
    def __init__(self, nb_classes):
        super().__init__(nb_classes_out=nb_classes)
        self.model = models.inception_v3(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.nb_classes_out, bias=True)


class ResNet50(Model):  # too long to compute
    def __init__(self, nb_classes):
        super().__init__(nb_classes_out=nb_classes)
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.nb_classes_out, bias=True)


class VGG19(Model):  # too long to compute
    def __init__(self, nb_classes):
        super().__init__(nb_classes_out=nb_classes)
        self.model = models.vgg19(pretrained=True)
        # self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.nb_classes_out, bias=True)

