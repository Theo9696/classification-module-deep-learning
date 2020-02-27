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


class Data:
    def __init__(self, list_data: list, batch_size: int):
        self.dataset_train = list_data[0]
        self.dataset_val = list_data[1]
        self.dataset_test = list_data[2]
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.dataset_train,
            batch_size=batch_size,
            shuffle=True)

    def is_data_ready_for_learning(self):
        return self.train_loader is not None


class DataImporter:
    def __init__(self, main_folder: str, batch_size: int = 100, split: SplitOptions = SplitOptions.SPLIT_ALL,
                 train_size: float = 0.8, test_size: float = 0.8, size_image_input_model: int = 256,
                 k_fold: bool = False, nb_chunk: int = 4):
        np.random.seed(42)
        trans = transforms.Compose([transforms.Resize([size_image_input_model, size_image_input_model]),
                                    transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        self.train_size = train_size

        if k_fold:
            self.data = self.create_cv(main_folder=main_folder, trans=trans, batch_size=batch_size,
                                       test_size=test_size, nb_chunk=nb_chunk)
        else:
            self.data = [Data(self.build_dataset(split=split,
                                                 main_folder=main_folder,
                                                 train_size=train_size, trans=trans,
                                                 size_image_input_model=size_image_input_model), batch_size=batch_size)]

        example = self.data[0]
        self.nb_train_samples = len(example.dataset_train)
        self.nb_val_samples = len(example.dataset_val)
        self.nb_test_samples = len(example.dataset_test)

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

        logger.info(f'total training batch number: {len(example.train_loader)}')

    @staticmethod
    def build_dataset(split: SplitOptions, main_folder: str, trans, train_size: float, size_image_input_model: int,
                      test_size: float = 0.8):
        if split.value is SplitOptions.SPLIT_ALL.value:
            train_data, val_data, test_data = DataImporter.split_data_all(main_folder, trans, train_size=train_size,
                                                                          test_size=test_size)
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

        return [dataset_train, dataset_val, dataset_test]

    @staticmethod
    def format_dataset(folder, trans, data):
        dataset = datasets.ImageFolder(root=folder, transform=trans)
        dataset.samples = data
        dataset.imgs = data
        return dataset

    @staticmethod
    def split_data_all(main_folder, trans, train_size: float, test_size: float):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, test_data = train_test_split(dataset_full.samples, train_size=test_size, shuffle=True)
        train_data, val_data = train_test_split(train_data, train_size=train_size, shuffle=True)
        return train_data, val_data, test_data

    @staticmethod
    def split_data_train(main_folder, trans, train_size: float):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, val_data = train_test_split(dataset_full.samples, train_size=train_size, shuffle=True)
        return train_data, val_data

    def imshow(self, tensor, title=None):
        img = tensor.cpu().clone()
        img = img.squeeze()
        plt.imshow(img, cmap='gray')
        if title is not None:
            plt.title(title)
        plt.pause(0.5)

    # def photo:
    #     plt.figure()
    #     for ii in range(10):
    #         imshow(train_set.data[ii, :, :], title='Example ({})'.format(train_set.targets[ii]))
    #     plt.close()

    @staticmethod
    def create_cv(main_folder: str, trans, batch_size: int, test_size: float, nb_chunk: int):
        return DataImporter.cross_validation_data(main_folder=main_folder, trans=trans,
                                                  batch_size=batch_size, test_size=test_size,
                                                  nb_chunk=nb_chunk)

    @staticmethod
    def cross_validation_data(main_folder: str, trans, nb_chunk: int, batch_size: int, test_size: float = 0.8):
        dataset_full = datasets.ImageFolder(root=main_folder, transform=trans)
        train_data, test_data = train_test_split(dataset_full.samples, train_size=test_size, shuffle=True)

        list_chunks = []
        for k in range(nb_chunk - 1):
            train_data, train_data_chunk = train_test_split(train_data, train_size=(1 - 1 / (nb_chunk - k)))
            list_chunks.append(train_data_chunk)
        list_chunks.append(train_data)

        list_of_trainings = []
        for k in range(nb_chunk):
            list_train = [list_chunks[(k + i) % nb_chunk] for i in range(nb_chunk - 1)]
            list_of_trainings.append(
                Data([DataImporter.format_dataset(main_folder, trans,
                                                  [element for chunk in list_train for element in chunk]),
                      DataImporter.format_dataset(main_folder, trans, list_chunks[(k + nb_chunk - 1) % nb_chunk]),
                      DataImporter.format_dataset(main_folder, trans, test_data)], batch_size=batch_size))
        return list_of_trainings


if __name__ == "__main__":
    list_train = DataImporter.cross_validation_data(main_folder="../data/plant-seedlings/train",
                                                    trans=transforms.Compose(
                                                        [transforms.Resize([256, 256]), transforms.ToTensor(),
                                                         transforms.Normalize((0.1307,), (0.3081,))]), nb_chunk=4,
                                                    test_size=0.9)
    print(list_train)
