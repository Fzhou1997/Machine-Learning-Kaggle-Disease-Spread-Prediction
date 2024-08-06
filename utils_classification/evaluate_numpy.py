from numpy import ndarray
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix


def evaluate_numpy(predicted: ndarray,
                   probability: ndarray,
                   actual: ndarray) -> tuple[float, float, float, float, float, ndarray]:
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    confusion = confusion_matrix(actual, predicted)
    roc_auc = roc_auc_score(actual, probability)
    return accuracy, precision, recall, f1, roc_auc, confusion
