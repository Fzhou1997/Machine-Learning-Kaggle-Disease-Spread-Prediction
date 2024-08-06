from numpy import ndarray
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix


def evaluate_tensor(predicted: Tensor,
                    probability: Tensor,
                    actual: Tensor) -> tuple[float, float, float, float, float, ndarray]:
    accuracy = Accuracy(task='binary')(predicted, actual).item()
    precision = Precision(task='binary')(predicted, actual).item()
    recall = Recall(task='binary')(predicted, actual).item()
    f1 = F1Score(task='binary')(predicted, actual).item()
    roc_auc = AUROC(task='binary')(probability, actual).item()
    confusion = ConfusionMatrix(task='binary')(predicted, actual).numpy()
    return accuracy, precision, recall, f1, roc_auc, confusion
