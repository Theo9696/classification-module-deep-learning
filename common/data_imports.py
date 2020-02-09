import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from common.logger import logger


class dataImporter:
    def __init__(self, train_path: str, test_path: str, batch_size: int = 100):
        trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        train_data = datasets.MNIST(train_path, train=True, transform=trans, download=True)
        test_data = datasets.MNIST(test_path, train=False, transform=trans, download=True)
        self.train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            batch_size=batch_size,
            shuffle=False)

        logger.info(f'total training batch number: {len(self.train_loader)}')
        logger.info(f'total testing batch number: {len(self.test_loader)}')

    def imshow(tensor, title=None):
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
