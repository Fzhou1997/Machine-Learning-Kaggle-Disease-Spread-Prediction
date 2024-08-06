import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from numpy import ndarray


def plot_confusion_matrix(confusion_matrix: ndarray,
                          classes: list[str] = ['Negative', 'Positive'],
                          title: str = 'Confusion matrix',
                          cmap: str = 'Blues') -> None:
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=classes)
    disp = disp.plot(cmap=cmap)
    disp.ax_.set_title(title)
    plt.show()
