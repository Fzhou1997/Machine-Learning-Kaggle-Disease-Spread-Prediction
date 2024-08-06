import torch
from torch import Tensor
from torch.utils.data import Dataset


class PopulationDataset(Dataset):
    def __init__(self, features: Tensor, labels: Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int | slice | Tensor) -> tuple[Tensor, Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.labels[idx]
