import torch
import torch.nn as nn
from torch.autograd import Variable
from common.dataImports import dataImporter
from common.logger import logger

NUM_CLASSES = 10
NUM_CONV_1 = 10  # try 32
NUM_CONV_2 = 20  # try 64
NUM_FC = 500  # try 1024


class TrainingGenerator:
    def __init__(self, model, data: dataImporter, number_epoch: int = 10, lr: float = -1, momentum: float = -1):
        self._model = model
        self._data = data
        self._number_epoch = number_epoch
        self._lr = lr if lr > 0 else None
        self._momentum = momentum if momentum > -1 else None

    def train(model, data: dataImporter, number_epoch: int = 10, lr: float = 0.05, momentum: float = 0.9):
        assert data is not None & data.is_data_ready_for_learning(), logger.error("Corrupted learning datas")

        # we use GPU if available, otherwise CPU
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # optimization hyperparameters
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
        loss_fn = nn.CrossEntropyLoss()

        # main loop (train+test)
        for epoch in range(number_epoch):
            # training
            model.train()  # mode "train" agit sur "dropout" ou "batchnorm"
            for batch_idx, (x, target) in enumerate(data.train_loader):
                optimizer.zero_grad()
                x, target = Variable(x).to(device), Variable(target).to(device)
                out = model(x)
                loss = loss_fn(out, target)
                loss.backward()  # backtracking automatic
                optimizer.step()
                if batch_idx % 100 == 0:
                    print('epoch {} batch {} [{}/{}] training loss: {}'.format(epoch, batch_idx, batch_idx * len(x),
                                                                               len(data.train_loader.dataset), loss.item()))
            #  testing
            model.eval()
            correct = 0
            with torch.no_grad():
                for batch_idx, (x, target) in enumerate(data.test_loader):
                    x, target = x.to(device), target.to(device)
                    out = model(x)
                    loss = loss_fn(out, target)
                    # _, prediction = torch.max(out.data, 1)
                    prediction = out.argmax(dim=1, keepdim=True)  # index of the max log-probability
                    correct += prediction.eq(target.view_as(prediction)).sum().item()
            taux_classif = 100. * correct / len(data.test_loader.dataset)
            print('Accuracy: {}/{} (tx {:.2f}%, err {:.2f}%)\n'.format(correct,
                                                                       len(data.test_loader.dataset), taux_classif,
                                                                       100. - taux_classif))
