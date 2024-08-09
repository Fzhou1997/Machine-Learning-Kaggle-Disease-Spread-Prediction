from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader


class Runner:
    """
    A class to run predictions on dataset features using a PyTorch model.

    Attributes:
        model (Module): The PyTorch model used for predictions.
    """

    def __init__(self, model: Module):
        """
        Initialize the Runner with a PyTorch model.

        Args:
            model (Module): The PyTorch model to be used for predictions.
        """
        self.model = model

    def predict(self, features: Dataset, batch_size: int = 32, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        """
        Predict binary labels for the given dataset features.

        Args:
            features (Dataset): The dataset containing features to be predicted.
            batch_size (int, optional): The batch size for the DataLoader. Default is 32.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.

        Returns:
            Tensor: The predicted binary labels.
        """
        self.model.to(device)
        self.model.eval()
        loader = DataLoader(features, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(device)
                predicted = self.model.predict(features)
                predicted = predicted.squeeze()
                predictions.append(predicted)
        return torch.cat(predictions, dim=0)

    def predict_proba(self, features: Dataset, batch_size: int = 32, device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
        """
        Predict probabilities for the given dataset features.

        Args:
            features (Dataset): The dataset containing features to be predicted.
            batch_size (int, optional): The batch size for the DataLoader. Default is 32.
            device (Literal['cpu', 'cuda', 'mps'], optional): The device to run the model on. Default is 'cpu'.

        Returns:
            Tensor: The predicted probabilities.
        """
        self.model.to(device)
        self.model.eval()
        loader = DataLoader(features, batch_size=batch_size, shuffle=False)
        predictions = []
        with torch.no_grad():
            for features, _ in loader:
                features = features.to(device)
                predicted = self.model(features)
                predicted = predicted.squeeze()
                predictions.append(predicted)
        return torch.cat(predictions, dim=0)