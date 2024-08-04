from typing import Self

import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, accuracy_score, confusion_matrix


class Evaluator:
    def __init__(self):
        self.predicted = None
        self.actual = None
        self.accuracy = None
        self.precision = None
        self.recall = None
        self.f1 = None
        self.roc_auc = None
        self.confusion_matrix = None

    def evaluate(self,
                 predicted: np.ndarray,
                 actual: np.ndarray) -> Self:
        self.predicted = predicted
        self.actual = actual

        return self
