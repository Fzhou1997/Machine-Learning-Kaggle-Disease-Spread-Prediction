from typing import Self

from numpy import ndarray
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler


class GaussianDiscriminantAnalysisRadialBasisFunction:
    def __init__(self):
        self.kernel = None
        self.model = None

    def fit(self,
            x: ndarray,
            y: ndarray,
            gamma: float = 1.0,
            random_state: int = 42) -> Self:
        self.kernel = RBFSampler(gamma=gamma, random_state=random_state)
        x_features = self.kernel.fit_transform(x)
        self.model = QuadraticDiscriminantAnalysis().fit(x_features, y)
        return self

    def predict(self, x: ndarray) -> ndarray:
        x_features = self.kernel.transform(x)
        return self.model.predict(x_features)

    def predict_proba(self, x: ndarray) -> ndarray:
        x_features = self.kernel.transform(x)
        return self.model.predict_proba(x_features)

    def score(self, x: ndarray, y: ndarray) -> float:
        x_features = self.kernel.transform(x)
        return self.model.score(x_features, y)
