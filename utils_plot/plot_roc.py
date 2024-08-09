import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.metrics import roc_curve, roc_auc_score

def plot_roc(probability: ndarray,
             actual: ndarray,
             title: str = "Receiver Operating Characteristic",
             figsize: tuple[float, float] = (8, 8)):
    """
    Plot a Receiver Operating Characteristic (ROC) curve using matplotlib.

    Args:
        probability (ndarray): Array containing the predicted probabilities.
        actual (ndarray): Array containing the actual class labels.
        title (str, optional): Title of the plot. Default is "Receiver Operating Characteristic".
        figsize (tuple[float, float], optional): Size of the figure. Default is (8, 8).

    Returns:
        None
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probability)
    auc = roc_auc_score(actual, probability)
    plt.figure(figsize=figsize)
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()