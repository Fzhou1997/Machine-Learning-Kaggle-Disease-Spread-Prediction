import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, Precision, Recall, AUROC, ROC, ConfusionMatrix, Metric

from utils_classification.models import ModelMultiLayerPerceptron


class TesterMultiLayerPerceptron:
    model: ModelMultiLayerPerceptron
    criterion: Module
    test_loader: DataLoader
    device: torch.device
    accuracy: Metric
    f1: Metric
    precision: Metric
    recall: Metric
    auroc: Metric
    roc: Metric
    confusion_matrix: Metric

    _loss: float
    _accuracy_score: float
    _precision_score: float
    _recall_score: float
    _f1_score: float
    _auroc_score: float
    _roc_curve: tuple[list[float], list[float], list[float]]
    _confusion_matrix_scores: list[list[int]]

    def __init__(self,
                 model: ModelMultiLayerPerceptron,
                 criterion: Module,
                 test_loader: DataLoader,
                 device: torch.device):
        self.model = model
        self.criterion = criterion
        self.test_loader = test_loader
        self.device = device
        self.accuracy = Accuracy(task='binary').to(device)
        self.f1 = F1Score(task='binary').to(device)
        self.precision = Precision(task='binary').to(device)
        self.recall = Recall(task='binary').to(device)
        self.auroc = AUROC(task='binary').to(device)
        self.roc = ROC(task='binary').to(device)
        self.confusion_matrix = ConfusionMatrix(task='binary').to(device)

        self._loss = 0
        self._accuracy_score = 0
        self._precision_score = 0
        self._recall_score = 0
        self._f1_score = 0
        self._auroc_score = 0
        self._roc_curve = ([], [], [])
        self._confusion_matrix_scores = []

    def test(self):
        self.model.eval()
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.auroc.reset()
        self.roc.reset()
        self.confusion_matrix.reset()

        with torch.no_grad():
            for features, label in self.test_loader:
                features, label = features.to(self.device), label.to(self.device)
                logits = self.model.forward(features)
                logits = logits.squeeze()
                loss = self.criterion(logits, label)
                self._loss += loss.item()
                predicted = self.model.predict(logits)
                classified = self.model.classify(predicted)
                classified = classified.long()
                true = label.long()
                self.accuracy.update(classified, true)
                self.f1.update(classified, true)
                self.precision.update(classified, true)
                self.recall.update(classified, true)
                self.auroc.update(classified, true)
                self.roc.update(classified, true)
                self.confusion_matrix.update(classified, true)

        self._loss /= len(self.test_loader)
        self._accuracy_score = self.accuracy.compute().cpu().item()
        self._precision_score = self.precision.compute().cpu().item()
        self._recall_score = self.recall.compute().cpu().item()
        self._f1_score = self.f1.compute().cpu().item()
        self._auroc_score = self.auroc.compute().cpu().item()
        fpr, tpr, thresholds = self.roc.compute()
        self._roc_curve = (fpr.cpu().tolist(), tpr.cpu().tolist(), thresholds.cpu().tolist())
        self._confusion_matrix_scores = self.confusion_matrix.compute().cpu().tolist()

    @property
    def loss(self) -> float:
        return self._loss

    @property
    def accuracy_score(self) -> float:
        return self._accuracy_score

    @property
    def precision_score(self) -> float:
        return self._precision_score

    @property
    def recall_score(self) -> float:
        return self._recall_score

    @property
    def f1_score(self) -> float:
        return self._f1_score

    @property
    def auroc_score(self) -> float:
        return self._auroc_score

    @property
    def roc_curve(self) -> tuple[list[float], list[float], list[float]]:
        return self._roc_curve

    @property
    def confusion_matrix_scores(self) -> list[list[int]]:
        return self._confusion_matrix_scores
