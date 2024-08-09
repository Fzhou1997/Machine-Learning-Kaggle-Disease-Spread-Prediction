from typing import Literal
from numpy import ndarray
from torch import Tensor
from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix

def evaluate_tensor(predicted: Tensor,
                    probability: Tensor,
                    actual: Tensor,
                    device: Literal['cpu', 'cuda', 'mps'] = 'cpu') -> tuple[float, float, float, float, float, ndarray]:
    """
    Evaluate various metrics for PyTorch tensors in a binary classification task.

    Args:
        predicted (Tensor): Tensor containing the predicted class labels.
        probability (Tensor): Tensor containing the predicted probabilities.
        actual (Tensor): Tensor containing the actual class labels.
        device (Literal['cpu', 'cuda', 'mps'], optional): Device to perform the evaluation on. Default is 'cpu'.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - accuracy (float): Accuracy of the predictions.
            - precision (float): Precision of the predictions.
            - recall (float): Recall of the predictions.
            - f1 (float): F1 score of the predictions.
            - roc_auc (float): Area Under the Receiver Operating Characteristic Curve (ROC AUC).
            - confusion (ndarray): Confusion matrix of the predictions.
    """
    accuracy = (Accuracy(task='binary').to(device))(predicted, actual).cpu().item()
    precision = (Precision(task='binary').to(device))(predicted, actual).cpu().item()
    recall = (Recall(task='binary').to(device))(predicted, actual).cpu().item()
    f1 = (F1Score(task='binary').to(device))(predicted, actual).cpu().item()
    roc_auc = (AUROC(task='binary').to(device))(probability, actual).cpu().item()
    confusion = (ConfusionMatrix(task='binary').to(device))(predicted, actual).cpu().numpy()
    return accuracy, precision, recall, f1, roc_auc, confusion