import torch

from utils_torch.data import DatasetFeatureTargetClassificationBinary


class DatasetMultiLayerPerceptron(DatasetFeatureTargetClassificationBinary):

    def __init__(self,
                 features: torch.Tensor,
                 targets: torch.Tensor) -> None:
        assert features.dim() == 2, f"Expected 2D features tensor, got {features.dim()}"
        assert targets.dim() == 1, f"Expected 1D targets tensor, got {targets.dim()}"
        assert features.shape[0] == targets.shape[
            0], f"Expected features and targets to have the same number of samples, got {features.shape[0]} and {targets.shape[0]}"
        self.features = features
        self.targets = targets

    @property
    def num_samples(self) -> int:
        return self.features.shape[0]

    @property
    def num_features(self) -> int:
        return self.features.shape[1]
