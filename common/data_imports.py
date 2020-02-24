import matplotlib.pyplot as plt
import torch
import torch.utils
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from common.logger import logger
import numpy as np
from enum import Enum
from data_saver.excel_actions import ParametersNames


class SplitOptions(Enum):
    SPLIT_ALL = 0
    SPLIT_TRAIN = 1
    NO_SPLIT = 2


class DataImporter:
    def __init__(self, main_folder: str, batch_size: int = 100, split: SplitOptions = SplitOptions.SPLIT_ALL,
                 train_size: float = 0.8, size_image_input_model: int = 256):

        np.random.seed(42)
        self.train_size = train_size
        self.dataset_train, self.dataset_val, self.dataset_test = self.build_dataset(split=split,
                                                                                     main_folder=main_folder,
                                                                                     train_size= train_size,
                                                                                     size_image_input_model=size_image_input_model)
        self.nb_train_samples = len(self.dataset_train)
        self.nb_val_samples = len(self.dataset_val)
        self.nb_test_samples = len(self.dataset_test)

        self.parameters_data_input = {
            ParametersNames.NB_TRAIN.value: self.nb_train_samples,
            ParametersNames.NB_VAL.value: self.nb_val_samples,
            ParametersNames.NB_TEST.value: self.nb_test_samples,
            ParametersNames.SIZE_IMAGE_INPUT_MODEL.value: size_image_input_model
        }

        logger.info(f"Number of train images: {self.nb_train_samples}")
        logger.info(f"Number of validation images: {self.nb_val_samples}")
        logger.info(f"Number of test images: {self.nb_test_samples}")

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

        logger.info(f'total training batch number: {len(self.train_loader)}')
        logger.info(f'total testing batch number: {len(self.test_loader)}')

    @staticmethod
    def build_dataset(split: SplitOptions, main_folder: str, train_size: float, size_image_input_model: int):
        trans = transforms.Compose([transforms.Resize([size_image_input_model, size_image_input_model]),
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

        if split.value is SplitOptions.SPLIT_ALL.value:
            train_data, val_data, test_data = DataImporter.split_data_all(main_folder, trans)
            dataset_train = DataImporter.format_dataset(main_folder, trans, train_data)
            dataset_val = DataImporter.format_dataset(main_folder, trans, val_data)
            dataset_test = DataImporter.format_dataset(main_folder, trans, test_data)

        elif split.value is SplitOptions.SPLIT_TRAIN.value:
            dataset_test = datasets.ImageFolder(root=main_folder + "/test", transform=trans)
            train_data, val_data = DataImporter.split_data_train(main_folder=main_folder + "/train", trans=trans,
                                                                 train_size=train_size)
            dataset_train = DataImporter.format_dataset(main_folder, trans, train_data)
            dataset_val = DataImporter.format_dataset(main_folder, trans, val_data)

        else:
            dataset_train = datasets.ImageFolder(root=main_folder + "/train", transform=trans)
            dataset_val = datasets.ImageFolder(root=main_folder + "/val", transform=trans)
            dataset_test = datasets.ImageFolder(root=main_folder + "/test", transform=trans)

        return dataset_train, dataset_val, dataset_test

    @staticmethod
    def format_dataset(folder, trans, data):
        dataset = datasets.ImageFolder(root=folder, transform=trans)
        dataset.samples = data
        dataset.imgs = data
        return dataset

    @staticmethod
    def split_data_all(main_folder, trans):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, test_data = train_test_split(dataset_full.samples)
        train_data, val_data = train_test_split(train_data)
        return train_data, val_data, test_data

    @staticmethod
    def split_data_train(main_folder, trans, train_size: float):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, val_data = train_test_split(dataset_full.samples, train_size=train_size)
        return train_data, val_data

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
