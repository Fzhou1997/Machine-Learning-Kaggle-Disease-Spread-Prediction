from typing import Literal

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from torchmetrics import Accuracy, Metric
from tqdm import tqdm

from .ModelMultiLayerPerceptron import ModelMultiLayerPerceptron


class TrainerMultiLayerPerceptron:
    model: ModelMultiLayerPerceptron
    criterion: Module
    optimizer: Optimizer
    train_loader: DataLoader
    eval_loader: DataLoader
    device: torch.device
    accuracy: Metric

    _num_epochs_trained: int
    _train_losses: list[float]
    _train_accuracies: list[float]
    _eval_losses: list[float]
    _eval_accuracies: list[float]

    def __init__(self,
                 model: ModelMultiLayerPerceptron,
                 criterion: Module,
                 optimizer: Optimizer,
                 train_loader: DataLoader,
                 eval_loader: DataLoader,
                 device: torch.device):

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.accuracy = Accuracy(task='binary')

        self._num_epochs_trained = 0
        self._train_losses = []
        self._train_accuracies = []
        self._eval_losses = []
        self._eval_accuracies = []

    def _train_one_epoch(self) -> float:
        running_loss = 0
        self.model.train()
        for features, label in self.train_loader:
            self.optimizer.zero_grad()
            features, label = features.to(self.device), label.to(self.device)
            predicted = self.model(features)
            predicted = predicted.squeeze()
            loss = self.criterion(predicted, label)
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()
        running_loss /= len(self.train_loader)
        return running_loss

    def _eval_one_epoch(self) -> float:
        loss = 0
        self.model.eval()
        with torch.no_grad():
            for features, label in self.eval_loader:
                features, label = features.to(self.device), label.to(self.device)
                predicted = self.model(features)
                predicted = predicted.squeeze()
                loss += self.criterion(predicted, label).item()
        loss /= len(self.eval_loader)
        return loss

    def train(self,
              num_epochs: int = 100,
              verbose: bool = True) -> None:
        best_eval_loss = float('inf')
        best_model_state = None
        best_model_epoch = -1
        for epoch in tqdm(range(num_epochs)):
            train_loss = self._train_one_epoch()
            eval_loss = self._eval_one_epoch()
            self.train_losses.append(train_loss)
            self.eval_losses.append(eval_loss)
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                best_model_state = self.model.state_dict()
                best_model_epoch = epoch
        print(
            f'Best model found at epoch {best_model_epoch} with evaluation loss: {best_eval_loss:.4f}')
        self.model.load_state_dict(best_model_state)

    def test(self,
             test_set: Dataset,
             batch_size: int = 32,
             device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        """
        Test the model on the given dataset.

        Args:
            test_set (Dataset): The dataset for testing.
            batch_size (int, optional): The batch size for the DataLoader. Default is 32.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.
        """
        self.device = device
        self.model.to(self.device)
        self.model.eval()
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
        loss = 0
        with torch.no_grad():
            for features, label in test_loader:
                features, label = features.to(self.device), label.to(self.device)
                predicted = self.model(features)
                predicted = predicted.squeeze()
                loss += self.criterion(predicted, label).item()
        loss /= len(test_loader)
        print(f'Test loss: {loss:.4f}')
