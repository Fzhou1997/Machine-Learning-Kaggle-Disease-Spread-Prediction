import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from numpy import ndarray

def plot_confusion_matrix(confusion_matrix: ndarray,
                          classes: list[str] = ['Negative', 'Positive'],
                          title: str = 'Confusion matrix',
                          cmap: str = 'Blues') -> None:
    """
    Plot a confusion matrix using matplotlib and sklearn.

    Args:
        confusion_matrix (ndarray): The confusion matrix to be plotted.
        classes (list[str], optional): List of class names to display on the axes. Default is ['Negative', 'Positive'].
        title (str, optional): Title of the plot. Default is 'Confusion matrix'.
        cmap (str, optional): Colormap to be used for the plot. Default is 'Blues'.

    Returns:
        None
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    disp = disp.plot(cmap=cmap)
    disp.ax_.set_title(title)
    plt.show()