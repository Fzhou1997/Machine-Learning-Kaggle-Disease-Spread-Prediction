from typing import Self

from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler


class GaussianDiscriminantAnalysisKernelFunction:
    def __init__(self,
                 kernel: RBFSampler | Nystroem | AdditiveChi2Sampler | SkewedChi2Sampler,
                 model: LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis):
        self.kernel = kernel
        self.model = model

    def fit(self, features: ndarray, labels: ndarray) -> Self:
        transformed_features = self.kernel.fit_transform(features)
        self.model = self.model.fit(transformed_features, labels)
        return self

    def predict(self, features: ndarray) -> ndarray:
        transformed_features = self.kernel.transform(features)
        return self.model.predict(transformed_features)

    def predict_proba(self, features: ndarray) -> ndarray:
        transformed_features = self.kernel.transform(features)
        return self.model.predict_proba(transformed_features)
