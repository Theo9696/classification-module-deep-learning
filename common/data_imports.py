import matplotlib.pyplot as plt
import torch
import torch.utils
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from common.logger import logger
import numpy as np


class DataImporter:
    def __init__(self, train_path: str, test_path: str, batch_size: int = 100):
        trans = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        # train_data = datasets.MNIST(train_path, train=True, transform=trans, download=True)
        # test_data = datasets.MNIST(test_path, train=False, transform=trans, download=True)

        # on définit les transformations à appliquer aux images du dataset
        # data_transforms = transforms.Compose([
        #     transforms.Resize([224, 224]),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=mean, std=std)
        # ])

        dataset_full = datasets.ImageFolder(root="data_beer", transform=trans)
        np.random.seed(42)
        train_data, test_data = train_test_split(dataset_full.samples)
        train_data, val_data = train_test_split(train_data)

        print("Nombre d'images de train : %i" % len(train_data))
        print("Nombre d'images de val : %i" % len(val_data))
        print("Nombre d'images de test : %i" % len(test_data))

        dataset_train = datasets.ImageFolder(root="data_beer", transform=trans)
        dataset_train.samples = train_data
        dataset_train.imgs = train_data

        dataset_val = datasets.ImageFolder(root="data_beer", transform=trans)
        dataset_val.samples = val_data
        dataset_val.imgs = val_data

        dataset_test = datasets.ImageFolder(root="data_beer", transform=trans)
        dataset_test.samples = test_data
        dataset_test.imgs = test_data

        # on définit les datasets et loaders pytorch à partir des listes d'images de train / val / test
        # dataset_train = datasets.ImageFolder(image_directory, data_transforms)
        # dataset_train.samples = samples_train
        # dataset_train.imgs = samples_train
        # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=dataset_train,
            batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=dataset_test,
            batch_size=batch_size,
            shuffle=False)

        # TODO: add eval

        logger.info(f'total training batch number: {len(self.train_loader)}')
        logger.info(f'total testing batch number: {len(self.test_loader)}')

    def imshow(self, tensor, title=None):
        img = tensor.cpu().clone()
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        if title is not None:
            plt.title(title)
        plt.pause(0.5)

    def is_data_ready_for_learning(self):
        return (self.train_loader is not None) & (self.test_loader is not None)

    # def photo:
    #     plt.figure()
    #     for ii in range(10):
    #         imshow(train_set.data[ii, :, :], title='Example ({})'.format(train_set.targets[ii]))
    #     plt.close()
