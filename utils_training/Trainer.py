from typing import Literal

import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class Trainer:
    """
    A class to train and evaluate a PyTorch model on dataset features.

    Attributes:
        model (Module): The PyTorch model to be trained.
        criterion (Module): The loss function.
        optimizer (Optimizer): The optimizer for training the model.
        train_loader (DataLoader): DataLoader for the training dataset.
        eval_loader (DataLoader): DataLoader for the evaluation dataset.
        train_losses (list): List to store training losses for each epoch.
        eval_losses (list): List to store evaluation losses for each epoch.
        device (str): The device to run the model on.
    """

    def __init__(self,
                 model: Module,
                 criterion: Module,
                 optimizer: Optimizer):
        """
        Initialize the Trainer with a model, criterion, and optimizer.

        Args:
            model (Module): The PyTorch model to be trained.
            criterion (Module): The loss function.
            optimizer (Optimizer): The optimizer for training the model.
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = None
        self.eval_loader = None
        self.train_losses = None
        self.eval_losses = None
        self.device = 'cpu'

    def _train_one_epoch(self) -> float:
        """
        Train the model for one epoch.

        Returns:
            float: The training loss for the epoch.
        """
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
        """
        Evaluate the model for one epoch.

        Returns:
            float: The evaluation loss for the epoch.
        """
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
              train_set: Dataset,
              eval_set: Dataset,
              batch_size: int = 32,
              num_epochs: int = 100,
              device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> None:
        """
        Train the model on the given dataset.

        Args:
            train_set (Dataset): The dataset for training.
            eval_set (Dataset): The dataset for evaluation.
            batch_size (int, optional): The batch size for the DataLoader. Default is 32.
            num_epochs (int, optional): The number of epochs to train. Default is 100.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.
        """
        self.train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        self.eval_loader = DataLoader(eval_set, batch_size=batch_size, shuffle=False)
        self.device = device
        self.model.to(self.device)
        self.train_losses = []
        self.eval_losses = []
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
        print(f'Best model found at epoch {best_model_epoch} with evaluation loss: {best_eval_loss:.4f}')
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