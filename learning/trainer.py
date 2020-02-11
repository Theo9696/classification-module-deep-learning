import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from common.data_imports import DataImporter
from common.logger import logger


class TrainingGenerator:
    def __init__(self, model: str, data: DataImporter, number_epoch: int = 10, lr: float = 0.05, momentum: float = -1,
                 print_val=True):
        self._model = model
        self._data = data
        self._number_epoch = number_epoch
        self._lr = lr if lr > 0 else 0.05
        self._momentum = momentum if momentum > -1 else None
        self.criterion = nn.CrossEntropyLoss()
        self.print_val = print_val

    def evaluate(self, model, dataset, device):
        avg_loss = 0.
        avg_accuracy = 0
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)
            n_correct = torch.sum(preds == labels)

            avg_loss += loss.item()
            avg_accuracy += n_correct

        return avg_loss / len(dataset), float(avg_accuracy) / len(dataset)

    def train(self):
        ts = time.time()
        assert (self._data is not None) & (self._data.is_data_ready_for_learning()), logger.error(
            "Corrupted learning datas")

        # we use GPU if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # optimization hyperparameters
        if self._momentum:
            optimizer = torch.optim.SGD(self._model.parameters, lr=self._lr, momentum=self._momentum)
        else:
            model = self._model
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()

        # main loop (train+test)
        for epoch in range(self._number_epoch):
            # training
            self._model.train(True)  # mode "train" agit sur "dropout" ou "batchnorm"
            for batch_idx, (x, target) in enumerate(self._data.train_loader):
                optimizer.zero_grad()
                x, target = Variable(x).to(device), Variable(target).to(device)
                out = self._model(x)
                loss = loss_fn(out, target)

                if self.print_val & (batch_idx % 5 == 0):
                    self._model.train(False)
                    loss_val, accuracy = self.evaluate(self._model, self._data.dataset_val, device)
                    self._model.train(True)
                    print(
                        "EPOCH {} | batch: {} loss train: {:1.4f}\t val {:1.4f}\tAcc: {:.1%}".format(epoch, batch_idx,
                                                                                                     loss.item(),
                                                                                                     loss_val,
                                                                                                     accuracy))
                loss.backward()  # backtracking automatic
                optimizer.step()
        logger.info(f"Training completed in {time.time() - ts} s")

    def test(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._model.train(False)
        loss_val, accuracy = self.evaluate(self._model, self._data.dataset_test, device)
        print(
            "TEST | loss val {:1.4f}\tAcc: {:.1%}".format(loss_val, accuracy))
