import os

import matplotlib.pyplot as plt
from numpy import ndarray
from sklearn.metrics import roc_curve, roc_auc_score


def plot_roc(probability: ndarray,
             actual: ndarray,
             width: float = 6,
             height: float = 6,
             title: str = "Receiver Operating Characteristic",
             xlabel: str = "False Positive Rate",
             ylabel: str = "True Positive Rate",
             output_path: str | bytes | os.PathLike[str] | os.PathLike[bytes] = None,
             output_file: str = None,
             show: bool = True) -> None:
    """
    Plot a Receiver Operating Characteristic (ROC) curve using matplotlib.

    Args:
        probability (ndarray): The probability of the positive class.
        actual (ndarray): The actual class labels.
        width (float, optional): The width of the plot. Defaults to 6.
        height (float, optional): The height of the plot. Defaults to 6.
        title (str, optional): The title of the plot. Defaults to "Receiver Operating Characteristic".
        xlabel (str, optional): The x-axis label. Defaults to "False Positive Rate".
        ylabel (str, optional): The y-axis label. Defaults to "True Positive Rate".
        output_path (str | bytes | os.PathLike[str] | os.PathLike[bytes], optional): The output path to save the plot. Defaults to None.
        output_file (str, optional): The output file name. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True.

    Returns:
        None
    """
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probability)
    auc = roc_auc_score(actual, probability)

    plt.figure(figsize=(width, height))
    plt.plot(false_positive_rate, true_positive_rate, label=f"ROC Curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    if show:
        plt.show()
