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
