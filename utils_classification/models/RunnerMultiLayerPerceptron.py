import torch
from torch.utils.data import DataLoader

from utils_classification.models import ModelMultiLayerPerceptron


class RunnerMultiLayerPerceptron:
    model: ModelMultiLayerPerceptron
    device: torch.device

    def __init__(self,
                 model: ModelMultiLayerPerceptron,
                 device: torch.device):
        self.model = model
        self.device = device

    def forward(self,
                inference_loader: DataLoader) -> torch.Tensor:
        logits = []
        for features in inference_loader:
            features = features.to(self.device)
            with torch.no_grad():
                logits_current = self.model.forward(features)
            logits.append(logits_current)
        logits = torch.cat(logits, dim=0)
        return logits

    def predict(self,
                inference_loader: DataLoader) -> torch.Tensor:
        predicted = []
        for features in inference_loader:
            features = features.to(self.device)
            with torch.no_grad():
                logits = self.model.forward(features)
                predicted_current = self.model.predict(logits)
            predicted.append(predicted_current)
        predicted = torch.cat(predicted, dim=0)
        return predicted

    def classify(self,
                 inference_loader: DataLoader) -> torch.Tensor:
        classified = []
        for features in inference_loader:
            features = features.to(self.device)
            with torch.no_grad():
                logits = self.model.forward(features)
                predicted = self.model.predict(logits)
                classified_current = self.model.classify(predicted)
            classified.append(classified_current)
        classified = torch.cat(classified, dim=0)
        return classified
