import os
from os import PathLike
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from numpy.typing import NDArray


def plot_confusion_matrix(confusion_matrix: list[list[int]] | NDArray[np.int_],
                          classes: Sequence[str] = None,
                          title: str = 'Confusion matrix',
                          width: int = 6,
                          height: int = 6,
                          output_path: str | bytes | PathLike[str] | PathLike[bytes] = None,
                          output_file: str = None,
                          show: bool = True) -> None:
    """
    Plot a confusion matrix using matplotlib and sklearn.

    Args:
        confusion_matrix (list[list[int] | NDArray[np.int_]]): The confusion matrix.
        classes (Sequence[str], optional): The class labels. Defaults to None.
        title (str, optional): The title of the plot. Defaults to 'Confusion matrix'.
        width (int, optional): The width of the plot. Defaults to 8.
        height (int, optional): The height of the plot. Defaults to 6.
        output_path (str | bytes | PathLike[str] | PathLike[bytes], optional): The output path to save the plot. Defaults to None.
        output_file (str, optional): The output file name. Defaults to None.
        show (bool, optional): Whether to show the plot. Defaults to True

    Returns:
        None
    """
    if isinstance(confusion_matrix, list):
        num_classes = len(confusion_matrix)
        for row in confusion_matrix:
            assert len(row) == num_classes, "The confusion matrix must be square."
    else:
        num_classes = confusion_matrix.shape[0]
        for row in confusion_matrix:
            assert row.shape[0] == num_classes, "The confusion matrix must be square."
    if classes is not None:
        assert len(classes) == num_classes, "The number of classes must be the same as the number of classes in the confusion matrix."
    elif num_classes == 2:
        classes = ['Negative', 'Positive']
    else:
        classes = range(num_classes)
    fig, ax = plt.subplots(figsize=(width, height))
    vmin = 0
    vmax = float(np.max(confusion_matrix))

    mask = np.eye(num_classes, dtype=bool)
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Reds",
                cbar=False,
                annot_kws={"size": 12},
                linewidths=0.5,
                ax=ax,
                mask=mask,
                square=True,
                linecolor="white",
                vmin=vmin,
                vmax=vmax)

    inverse_mask = ~mask
    sns.heatmap(confusion_matrix,
                annot=True,
                fmt="d",
                cmap="Blues",
                cbar=False,
                annot_kws={"size": 12},
                linewidths=0.5,
                ax=ax,
                mask=inverse_mask,
                square=True,
                linecolor="white",
                vmin=vmin,
                vmax=vmax)

    ax.set_xticklabels(classes, rotation=0)
    ax.set_yticklabels(classes, rotation=90)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.title(title)

    if output_file is not None:
        if output_path is None:
            output_path = os.getcwd()
        plt.savefig(os.path.join(output_path, output_file))

    if show:
        plt.show()
