import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from common.data_imports import DataImporter
from common.logger import logger
from data_saver.excel_actions import SheetSaver, SheetNames, ParametersNames
from models.Model import Model


class TrainingGenerator:
    def __init__(self, model: Model, data: DataImporter, number_epoch: int = 10, lr: float = 0.05, momentum: float = -1,
                 print_intermediate_perf=True, save_performances=True, sheet_name: str = "", location_to_save: str = "",
                 parameters_data_input: dict = None):
        self._model = model.model
        self._data = data
        self._number_epoch = number_epoch
        self._lr = lr if lr > 0 else 0.05
        self._momentum = momentum if momentum > -1 else None
        self.criterion = nn.CrossEntropyLoss()
        self.print_val = print_intermediate_perf
        self.save_val = save_performances
        self.sheet_saver = SheetSaver(location_to_save)
        self.dict_to_save = {SheetNames.PARAMETERS.value: {ParametersNames.MODEL.value: type(self._model).__name__,
                                                           ParametersNames.NB_EPOCH.value: self._number_epoch,
                                                           ParametersNames.LEARNING_RATE.value: self._lr,
                                                           ParametersNames.MOMENTUM.value: self._momentum},
                             SheetNames.PARAMETERS_MODELS.value: model.get_parameters(),
                             SheetNames.TRAIN_ERROR.value: [],
                             SheetNames.LOSS_FUNCTION.value: [],
                             SheetNames.VAL_ERROR.value: [],
                             SheetNames.TEST_ERROR.value: []}
        self.sheet_name = sheet_name
        for element in parameters_data_input:
            self.dict_to_save[SheetNames.PARAMETERS_MODELS.value][element] = parameters_data_input[element]

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
                    self.show_score(epoch=epoch, batch_idx=batch_idx, item=loss.item(), device=device)
                loss.backward()  # backtracking automatic
                optimizer.step()

            self.show_score(epoch=epoch, item=loss.item(), device=device)
        time_to_fit = time.time() - ts
        self.dict_to_save[SheetNames.PARAMETERS_MODELS.value][ParametersNames.TIME.value] = time_to_fit
        logger.info(f"Training completed in {time_to_fit} s")

    def show_score(self, epoch: int, item, device, batch_idx: int = None):
        self._model.train(False)
        loss_val, accuracy = self.evaluate(self._model, self._data.dataset_val, device)
        self._model.train(True)
        message = f"loss train: {round(item, 3)} val: {round(loss_val, 3)} Acc: {accuracy * 100}%"
        if batch_idx is not None:
            message = f"EPOCH {epoch} | batch: {batch_idx} " + message
        else:
            message = f"EPOCH {epoch} | " + message
        logger.info(message)

        if self.save_val & (batch_idx is None):
            self.dict_to_save[SheetNames.TRAIN_ERROR.value].append((epoch + 1, item))
            self.dict_to_save[SheetNames.LOSS_FUNCTION.value].append((epoch + 1, loss_val))
            self.dict_to_save[SheetNames.VAL_ERROR.value].append((epoch + 1, accuracy))

    def test(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self._model.train(False)
        loss_val, accuracy = self.evaluate(self._model, self._data.dataset_test, device)
        logger.info(f"TEST | loss val: {loss_val} Acc: {accuracy}%")

        if self.save_val:
            self.dict_to_save[SheetNames.TEST_ERROR.value].append((self._number_epoch, accuracy))

            self.sheet_saver.write_dic(dic=self.dict_to_save, sheet_name=self.sheet_name)
