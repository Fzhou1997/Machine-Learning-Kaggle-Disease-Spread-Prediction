from numpy import ndarray
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix

def evaluate_numpy(predicted: ndarray,
                   probability: ndarray,
                   actual: ndarray) -> tuple[float, float, float, float, float, ndarray]:
    """
    Evaluate various metrics for numpy arrays in a binary classification task.

    Args:
        predicted (ndarray): Array containing the predicted class labels.
        probability (ndarray): Array containing the predicted probabilities.
        actual (ndarray): Array containing the actual class labels.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - accuracy (float): Accuracy of the predictions.
            - precision (float): Precision of the predictions.
            - recall (float): Recall of the predictions.
            - f1 (float): F1 score of the predictions.
            - roc_auc (float): Area Under the Receiver Operating Characteristic Curve (ROC AUC).
            - confusion (ndarray): Confusion matrix of the predictions.
    """
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted)
    recall = recall_score(actual, predicted)
    f1 = f1_score(actual, predicted)
    confusion = confusion_matrix(actual, predicted)
    roc_auc = roc_auc_score(actual, probability)
    return accuracy, precision, recall, f1, roc_auc, confusion