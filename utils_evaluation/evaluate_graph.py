from torchmetrics import Accuracy, Precision, Recall, F1Score, AUROC, ConfusionMatrix

def evaluate_graph_metrics(predicted, probabilities, actual, mask, device='cuda'):
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