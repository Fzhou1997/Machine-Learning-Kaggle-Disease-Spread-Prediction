import torch
from torch import Tensor
from torch.utils.data import Dataset

class PopulationDataset(Dataset):
    """
    A custom dataset for population data, inheriting from PyTorch's Dataset class.

    Args:
        features (Tensor): A tensor containing the features of the dataset.
        labels (Tensor): A tensor containing the labels of the dataset.
    """
    def __init__(self, features: Tensor, labels: Tensor) -> None:
        self.features = features
        self.labels = labels

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The number of samples.
        """
        return len(self.features)

    def __getitem__(self, idx: int | slice | Tensor) -> tuple[Tensor, Tensor]:
        """
        Retrieves a sample from the dataset at the specified index.

        Args:
            idx (int | slice | Tensor): The index or indices of the sample(s) to retrieve.

        Returns:
            tuple[Tensor, Tensor]: A tuple containing the features and labels of the sample(s).
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.features[idx], self.labels[idx]