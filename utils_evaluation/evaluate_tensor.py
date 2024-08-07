from typing import Literal

from numpy import ndarray
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix


def evaluate_tensor(predicted: Tensor,
                    probability: Tensor,
                    actual: Tensor,
                    device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> tuple[float, float, float, float, float, ndarray]:
    accuracy = (Accuracy(task='binary').to(device))(predicted, actual).cpu().item()
    precision = (Precision(task='binary').to(device))(predicted, actual).cpu().item()
    recall = (Recall(task='binary').to(device))(predicted, actual).cpu().item()
    f1 = (F1Score(task='binary').to(device))(predicted, actual).cpu().item()
    roc_auc = (AUROC(task='binary').to(device))(probability, actual).cpu().item()
    confusion = (ConfusionMatrix(task='binary').to(device))(predicted, actual).cpu().numpy()
    return accuracy, precision, recall, f1, roc_auc, confusion
