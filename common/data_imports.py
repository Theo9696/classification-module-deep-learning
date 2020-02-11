import matplotlib.pyplot as plt
import torch
import torch.utils
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from common.logger import logger
import numpy as np


class DataImporter:
    def __init__(self, main_folder: str, batch_size: int = 100, split=True):
        trans = transforms.Compose([transforms.Resize([256, 256]),
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        np.random.seed(42)

        if split:
            train_data, val_data, test_data = self.split_data(main_folder, trans)
            self.dataset_train = self.format_dataset(main_folder, trans, train_data)
            self.dataset_val = self.format_dataset(main_folder, trans, val_data)
            self.dataset_test = self.format_dataset(main_folder, trans, test_data)
        else:
            self.dataset_train = datasets.ImageFolder(root=main_folder + "/train", transform=trans)
            self.dataset_val = datasets.ImageFolder(root=main_folder + "/val", transform=trans)
            self.dataset_test = datasets.ImageFolder(root=main_folder + "/test", transform=trans)

        print("Nombre d'images de train : %i" % len(self.dataset_train))
        print("Nombre d'images de val : %i" % len(self.dataset_val))
        print("Nombre d'images de test : %i" % len(self.dataset_test))

        # on définit les datasets et loaders pytorch à partir des listes d'images de train / val / test
        # dataset_train = datasets.ImageFolder(image_directory, data_transforms)
        # dataset_train.samples = samples_train
        # dataset_train.imgs = samples_train
        # loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True, num_workers=4)

        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_test,
            batch_size=batch_size,
            shuffle=False)

        # TODO: add eval

        logger.info(f'total training batch number: {len(self.train_loader)}')
        logger.info(f'total testing batch number: {len(self.test_loader)}')

    @staticmethod
    def format_dataset(folder, trans, data):
        dataset = datasets.ImageFolder(root=folder, transform=trans)
        dataset.samples = data
        dataset.imgs = data
        return dataset

    @staticmethod
    def split_data(main_folder, trans):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, test_data = train_test_split(dataset_full.samples)
        train_data, val_data = train_test_split(train_data)
        return train_data, val_data, test_data

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
