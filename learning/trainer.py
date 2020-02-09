import torch
import torch.nn as nn
from torch.autograd import Variable
from common.data_imports import dataImporter
from common.logger import logger


class TrainingGenerator:
    def __init__(self, model, data: dataImporter, number_epoch: int = 10, lr: float = 0.05, momentum: float = -1):
        self._model = model
        self._data = data
        self._number_epoch = number_epoch
        self._lr = lr if lr > 0 else 0.05
        self._momentum = momentum if momentum > -1 else None

    def train(self):
        assert (self._data is not None) & (self._data.is_data_ready_for_learning()), logger.error("Corrupted learning datas")

        # we use GPU if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._model.to(device)

        # optimization hyperparameters
        if self._momentum:
            optimizer = torch.optim.SGD(self._model.parameters(), lr=self._lr, momentum=self._momentum)
        else:
            model = self._model
            optimizer = torch.optim.SGD(model.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()

        # main loop (train+test)
        for epoch in range(self._number_epoch):
            # training
            self._model.train()  # mode "train" agit sur "dropout" ou "batchnorm"
            for batch_idx, (x, target) in enumerate(self._data.train_loader):
                optimizer.zero_grad()
                x, target = Variable(x).to(device), Variable(target).to(device)
                out = self._model(x)
                loss = loss_fn(out, target)
                loss.backward()  # backtracking automatic
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch, batch_idx, batch_idx * len(x),
                                                                               len(self._data.train_loader.dataset), loss.item()))
            #  testing
            self._model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(self._data.test_loader):
                    x, target = x.to(device), target.to(device)
                    out = self._model(x)
                    loss = loss_fn(out, target)
                    # _, prediction = torch.max(out.data, 1)
                    prediction = out.argmax(dim=1, keepdim=True)  # index of the max log-probability
                    correct += prediction.eq(target.view_as(prediction)).sum().item()
            taux_classif = 100. * correct / len(self._data.test_loader.dataset)
            print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
                                                                       len(self._data.test_loader.dataset), taux_classif,
                                                                       100. - taux_classif))
