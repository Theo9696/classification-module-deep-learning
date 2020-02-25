from torch import nn
from torchvision import models
from models.Model import Model


class Resnet(Model):
    def __init__(self, nb_classes):
        super().__init__(nb_classes_out=nb_classes)
        self.model = models.resnet18(pretrained=True)
        self.model.fc = nn.Linear(in_features=self.model.fc.in_features, out_features=self.nb_classes_out, bias=True)

        # cette fois on veut updater tous les paramètres
        # NB: il serait possible de ne sélectionner que quelques couches
        #     (plutôt parmi les "dernières", proches de la loss)
        #    Exemple (dans ce cas, oter la suite "params_to_update = resnet.parameters()"):
        # list_of_layers_to_finetune=['fc.weight','fc.bias','layer4.1.conv2.weight','layer4.1.bn2.bias','layer4.1.bn2.weight']
        # params_to_update=[]
        # for name,param in resnet.named_parameters():
        #     if name in list_of_layers_to_finetune:
        #         print("fine tune ",name)
        #         params_to_update.append(param)
        #         param.requires_grad = True
        #     else:
        #         param.requires_grad = False
