from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader


class Runner:
    def __init__(self,
                 model: Module):
        self.model = model

    def predict(self,
                features: Dataset,
                batch_size: int = 32,
                device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
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

    def predict_proba(self,
                      features: Dataset,
                      batch_size: int = 32,
                      device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> Tensor:
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
