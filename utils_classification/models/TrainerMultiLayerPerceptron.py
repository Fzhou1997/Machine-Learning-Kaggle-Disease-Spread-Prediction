import copy

import numpy as np
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
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

    def _train_one_epoch(self) -> None:
        running_loss = 0
        self.accuracy.reset()
        self.model.train()
        for features, label in self.train_loader:
            self.optimizer.zero_grad()
            features, label = features.to(self.device), label.to(self.device)
            logits = self.model.forward(features)
            logits = logits.squeeze()
            loss = self.criterion(logits, label)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            predicted = self.model.predict(logits)
            classified = self.model.classify(predicted)
            classified = classified.long()
            true = label.long()
            self.accuracy.update(classified, true)
        running_loss /= len(self.train_loader)
        accuracy = self.accuracy.compute().cpu().item()
        self._num_epochs_trained += 1
        self._train_losses.append(running_loss)
        self._train_accuracies.append(accuracy)

    def _eval_one_epoch(self) -> None:
        loss = 0
        self.accuracy.reset()
        self.model.eval()
        with torch.no_grad():
            for features, label in self.eval_loader:
                features, label = features.to(self.device), label.to(self.device)
                logits = self.model(features)
                logits = logits.squeeze()
                loss += self.criterion(logits, label).item()
                predicted = self.model.predict(logits)
                classified = self.model.classify(predicted)
                classified = classified.long()
                true = label.long()
                self.accuracy.update(classified, true)
        loss /= len(self.eval_loader)
        accuracy = self.accuracy.compute().cpu().item()
        self._eval_losses.append(loss)
        self._eval_accuracies.append(accuracy)

    def train(self,
              num_epochs: int = 100,
              verbose: bool = True) -> None:
        best_epoch = -1
        best_loss = float('inf')
        best_state_dict = None
        for epoch in (range(num_epochs) if not verbose else tqdm(range(num_epochs))):
            self._train_one_epoch()
            self._eval_one_epoch()
            train_loss = self._train_losses[-1]
            train_accuracy = self._train_accuracies[-1]
            eval_loss = self._eval_losses[-1]
            eval_accuracy = self._eval_accuracies[-1]
            if verbose:
                print(f'Epoch {epoch + 1}/{num_epochs}')
                print(f'Train loss: {train_loss:.4f}, Train accuracy: {train_accuracy:.4f}')
                print(f'Eval loss: {eval_loss:.4f}, Eval accuracy: {eval_accuracy:.4f}')
            if eval_loss < best_loss:
                if verbose:
                    print(f'Eval loss decreased: {best_loss:.4f} -> {eval_loss:.4f}. Saving model...')
                best_epoch = epoch
                best_loss = eval_loss
                best_state_dict = copy.deepcopy(self.model.state_dict())
            if verbose:
                print("==" * 25)
        if verbose:
            print(f'Best model found at epoch {best_epoch + 1}')
        self.model.load_state_dict(best_state_dict)

    @property
    def num_epochs_trained(self) -> int:
        return self._num_epochs_trained

    @property
    def train_losses(self) -> list[float]:
        return self._train_losses

    @property
    def train_accuracies(self) -> list[float]:
        return self._train_accuracies

    @property
    def best_train_loss(self) -> float:
        return min(self._train_losses)

    @property
    def best_train_accuracy(self) -> float:
        return max(self._train_accuracies)

    @property
    def best_train_loss_epoch(self) -> int:
        return np.argmin(self._train_losses)

    @property
    def best_train_accuracy_epoch(self) -> int:
        return np.argmax(self._train_accuracies)

    @property
    def eval_losses(self) -> list[float]:
        return self._eval_losses

    @property
    def eval_accuracies(self) -> list[float]:
        return self._eval_accuracies

    @property
    def best_eval_loss(self) -> float:
        return min(self._eval_losses)

    @property
    def best_eval_accuracy(self) -> float:
        return max(self._eval_accuracies)

    @property
    def best_eval_loss_epoch(self) -> int:
        return np.argmin(self._eval_losses)

    @property
    def best_eval_accuracy_epoch(self) -> int:
        return np.argmax(self._eval_accuracies)
    
    @property
    def learning_rate(self) -> float:
        return self.optimizer.param_groups[0]['lr']
