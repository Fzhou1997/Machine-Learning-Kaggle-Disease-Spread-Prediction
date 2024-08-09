from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix

def evaluate_graph_metrics(predicted, probabilities, actual, mask, device='cuda'):
    """
    Evaluate graph metrics for binary classification.

    Args:
        predicted (torch.Tensor): Tensor containing the predicted class labels.
        probabilities (torch.Tensor): Tensor containing the predicted probabilities.
        actual (torch.Tensor): Tensor containing the actual class labels.
        mask (torch.Tensor): Boolean tensor to mask the relevant data points.
        device (str, optional): Device to perform the evaluation on. Default is 'cuda'.

    Returns:
        tuple: A tuple containing the following evaluation metrics:
            - accuracy (float): Accuracy of the predictions.
            - precision (float): Precision of the predictions.
            - recall (float): Recall of the predictions.
            - f1 (float): F1 score of the predictions.
            - auc_roc (float): Area Under the Receiver Operating Characteristic Curve (AUROC).
            - confusion (numpy.ndarray): Confusion matrix of the predictions.
    """
    predicted = predicted[mask].to(device)
    probabilities = probabilities[mask].to(device)
    actual = actual[mask].to(device)

    accuracy_metric = Accuracy(task='binary').to(device)
    precision_metric = Precision(task='binary').to(device)
    recall_metric = Recall(task='binary').to(device)
    f1_metric = F1Score(task='binary').to(device)
    auc_roc_metric = AUROC(task='binary').to(device)

    accuracy = accuracy_metric(predicted, actual).item()
    precision = precision_metric(predicted, actual).item()
    recall = recall_metric(predicted, actual).item()
    f1 = f1_metric(predicted, actual).item()
    auc_roc = auc_roc_metric(probabilities, actual).item()
    confusion = ConfusionMatrix(task='binary').to(device)(predicted, actual).cpu().numpy()

    return accuracy, precision, recall, f1, auc_roc, confusion