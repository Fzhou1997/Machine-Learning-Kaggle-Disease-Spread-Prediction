import matplotlib.pyplot as plt


def plot_roc(false_positive_rate: list[float],
             true_positive_rate: list[float],
             auc: float,
             title: str = None,
             figsize: tuple[float, float] = (12, 8)):
    plt.figure(figsize=figsize)
    plt.plot(false_positive_rate, true_positive_rate, label=f'AUC = {auc:.2f}')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)
    plt.legend()
    plt.show()
