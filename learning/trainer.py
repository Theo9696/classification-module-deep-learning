import time

import torch
import torch.nn as nn
from torch.autograd import Variable
from common.data_imports import DataImporter
from common.logger import logger
from data_saver.excel_actions import SheetSaver, SheetNames, ParametersNames, Result, TrainingResult
from models.Model import Model
import numpy as np


class TrainingGenerator:
    def __init__(self, model: Model, data: DataImporter, number_epoch: int = 10, lr: float = 0.05, momentum: float = -1,
                 print_intermediate_perf=True, save_performances=True, sheet_name: str = "", location_to_save: str = "",
                 parameters_data_input: dict = None):
        self._model = model.model
        self._model_info = model
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
                             SheetNames.TRAINING.value: {
                                 TrainingResult.ACCURACY.value: [],
                                 TrainingResult.LOSS_TRAIN.value: [],
                                 TrainingResult.LOSS_VAL.value: [],
                                 TrainingResult.TP.value: [],
                                 TrainingResult.FP.value: [],
                                 TrainingResult.FN.value: [],
                                 TrainingResult.TN.value: [],
                                 TrainingResult.RECALL.value: [],
                                 TrainingResult.PRECISION.value: [],
                                 TrainingResult.CONFUSION_MATRIX.value: []
                             },
                             SheetNames.RESULT.value: {}}
        self.sheet_name = sheet_name
        for element in parameters_data_input:
            self.dict_to_save[SheetNames.PARAMETERS_MODELS.value][element] = parameters_data_input[element]

    def evaluate(self, model, dataset, device):
        avg_loss = 0.
        avg_accuracy = 0
        is_binary_problem = self._model_info.nb_classes_out == 2
        results = {Result.FN.value: 0, Result.FP.value: 0, Result.TN.value: 0,
                   Result.TP.value: 0} if is_binary_problem else {}
        loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)
        for data in loader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = self.criterion(outputs, labels)
            _, preds = torch.max(outputs, 1)

            if is_binary_problem:
                confusion = TrainingGenerator.confusion(prediction=preds, truth=labels, device=device)
                results[Result.TP.value] += confusion[0]
                results[Result.FP.value] += confusion[1]
                results[Result.TN.value] += confusion[2]
                results[Result.FN.value] += confusion[3]
                results[Result.CONFUSION_MATRIX.value] = str(
                    np.array([[results[Result.TP.value], results[Result.FP.value]],
                              [results[Result.FN.value], results[Result.TN.value]]]))

            n_correct = torch.sum(preds == labels)

            avg_loss += loss.item()
            avg_accuracy += n_correct

        if is_binary_problem:
            results[Result.PRECISION.value] = results[Result.TP.value] / (
                        results[Result.TP.value] + results[Result.FP.value])
            results[Result.RECALL.value] = results[Result.TP.value] / (
                        results[Result.TP.value] + results[Result.FN.value])
            results[Result.ACCURACY.value] = float(avg_accuracy) / len(dataset)
            results[Result.LOSS.value] = avg_loss / len(dataset)

        return results

    def train(self):
        ts = time.time()
        assert (self._data is not None) & (self._data.is_data_ready_for_learning()), \
            logger.error("Corrupted learning datas")

        # we use GPU if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # optimization hyperparameters
        if self._momentum:
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, momentum=self._momentum)
        else:
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
        results = self.evaluate(self._model, self._data.dataset_val, device)
        self.print_results(is_val=True, results=results, batch_idx=batch_idx, item=item, epoch=epoch)
        self._model.train(True)

    def test(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        results = self.evaluate(self._model, self._data.dataset_test, device)
        self.print_results(is_test=True, results=results)
        self._model.train(False)

    def print_results(self, results: dict, is_test: bool = False, is_val: bool = False, batch_idx: int = None,
                      item=None, epoch: int = None):
        loss_val = results[Result.LOSS.value]
        accuracy = round(results[Result.ACCURACY.value], 4)
        precision = round(results[Result.PRECISION.value], 4)
        recall = round(results[Result.RECALL.value], 4)
        confusion_matrix = np.array([[results[Result.TP.value], results[Result.FP.value]],
                                     [results[Result.FN.value], results[Result.TN.value]]])

        if is_val & (epoch is not None):
            self.print_save_val_results(batch_idx=batch_idx, epoch=epoch, accuracy=accuracy, loss_val=loss_val,
                                        item=item, recall=recall, precision=precision,
                                        confusion_matrix=confusion_matrix)

        if is_test:
            self.print_save_test_results(loss_val=loss_val, accuracy=accuracy, precision=precision, recall=recall,
                                         confusion_matrix=confusion_matrix)
            if self.save_val:
                self.dict_to_save[SheetNames.RESULT.value] = results
                self.sheet_saver.write_dic(dictionary=self.dict_to_save, sheet_name=self.sheet_name)

    @staticmethod
    def print_save_test_results(loss_val: float, accuracy: float, precision: float, recall: float,
                                confusion_matrix):
        message = f"TEST | loss val: {loss_val} Acc: {accuracy * 100}%"

        if (precision is not None) & (recall is not None) & (confusion_matrix is not None):
            message += f" | precision: {precision}  recall: {recall}  confusion matrix: \n {str(confusion_matrix)}"

        logger.info(message)

    def print_save_val_results(self, batch_idx: int, item, loss_val: float, accuracy: float, epoch: int,
                               precision: float, recall: float, confusion_matrix):
        message = f"loss train: {round(item, 3)} val: {round(loss_val, 3)} Acc: {accuracy * 100}%"
        if batch_idx is not None:
            message = f"EPOCH {epoch} | batch: {batch_idx} " + message
        else:
            message = f"EPOCH {epoch} | " + message

        if (precision is not None) & (recall is not None) & (confusion_matrix is not None):
            message += f" | precision: {precision}  recall: {recall} | confusion matrix: \n {str(confusion_matrix)}"

        logger.info(message)

        if self.save_val & (batch_idx is None):
            training_results = self.dict_to_save[SheetNames.TRAINING.value]
            training_results[TrainingResult.LOSS_TRAIN.value].append((epoch + 1, item))
            training_results[TrainingResult.LOSS_VAL.value].append((epoch + 1, loss_val))
            training_results[TrainingResult.ACCURACY.value].append((epoch + 1, accuracy))
            training_results[TrainingResult.PRECISION.value].append((epoch + 1, precision))
            training_results[TrainingResult.RECALL.value].append((epoch + 1, recall))
            training_results[TrainingResult.CONFUSION_MATRIX.value].append((epoch + 1, str(confusion_matrix)))
            training_results[TrainingResult.TP.value].append((epoch + 1, int(confusion_matrix[0][0])))
            training_results[TrainingResult.FP.value].append((epoch + 1, int(confusion_matrix[0][1])))
            training_results[TrainingResult.FN.value].append((epoch + 1, int(confusion_matrix[1][0])))
            training_results[TrainingResult.TN.value].append((epoch + 1, int(confusion_matrix[1][1])))

    @staticmethod
    def confusion(prediction: torch.Tensor, truth: torch.Tensor, device):
        """ Returns the confusion matrix for the values in the `prediction` and `truth`
        tensors, i.e. the amount of positions where the values of `prediction`
        and `truth` are
        - 1 and 1 (True Positive)
        - 1 and 0 (False Positive)
        - 0 and 0 (True Negative)
        - 0 and 1 (False Negative)
        """

        prediction_shifted = prediction + torch.tensor([1] * len(prediction), device=device)
        truth_shifted = truth + torch.tensor([1 / 2] * len(truth), device=device)  # 0 -> 1/2
        confusion_vector = prediction_shifted / truth_shifted

        # Element-wise division of the 2 tensors returns a new tensor which holds a
        # unique value for each case:
        #   4/3     where prediction is 1 (2) and truth is 1 (3/2) (True Positive)
        #    4      where prediction is 1 (2) and truth is 0 (1/2) (False Positive)
        #    2      where prediction is 0 (1) and truth is 0 (1/2) (True Negative)
        #   2/3     where prediction is 0 (1) and truth is 1 (3/2) (False Negative)

        true_positives = torch.sum(confusion_vector == 4 / 3).item()
        false_positives = torch.sum(confusion_vector == 4).item()
        true_negatives = torch.sum(confusion_vector == 2).item()
        false_negatives = torch.sum(confusion_vector == 2 / 3).item()

        return true_positives, false_positives, true_negatives, false_negatives
