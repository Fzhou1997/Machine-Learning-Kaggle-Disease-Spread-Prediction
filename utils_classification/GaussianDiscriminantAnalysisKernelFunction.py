from typing import Self
from numpy import ndarray
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.kernel_approximation import RBFSampler, Nystroem, AdditiveChi2Sampler, SkewedChi2Sampler

class GaussianDiscriminantAnalysisKernelFunction:
    """
    A class to perform Gaussian Discriminant Analysis with kernel functions.

    Attributes:
    ----------
    kernel : RBFSampler | Nystroem | AdditiveChi2Sampler | SkewedChi2Sampler
        The kernel function to transform the features.
    model : LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis
        The discriminant analysis model to fit the transformed features.
    """

    def __init__(self,
                 kernel: RBFSampler | Nystroem | AdditiveChi2Sampler | SkewedChi2Sampler,
                 model: LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis):
        """
        Initializes the GaussianDiscriminantAnalysisKernelFunction with a kernel and a model.

        Parameters:
        ----------
        kernel : RBFSampler | Nystroem | AdditiveChi2Sampler | SkewedChi2Sampler
            The kernel function to transform the features.
        model : LinearDiscriminantAnalysis | QuadraticDiscriminantAnalysis
            The discriminant analysis model to fit the transformed features.
        """
        self.kernel = kernel
        self.model = model

    def fit(self, features: ndarray, labels: ndarray) -> Self:
        """
        Fits the model using the transformed features and labels.

        Parameters:
        ----------
        features : ndarray
            The input features to be transformed and used for fitting the model.
        labels : ndarray
            The target labels for the input features.

        Returns:
        -------
        Self
            The instance of the class.
        """
        transformed_features = self.kernel.fit_transform(features)
        self.model = self.model.fit(transformed_features, labels)
        return self

    def predict(self, features: ndarray) -> ndarray:
        """
        Predicts the labels for the given features using the fitted model.

        Parameters:
        ----------
        features : ndarray
            The input features to be transformed and used for prediction.

        Returns:
        -------
        ndarray
            The predicted labels.
        """
        transformed_features = self.kernel.transform(features)
        return self.model.predict(transformed_features)

    def predict_proba(self, features: ndarray) -> ndarray:
        """
        Predicts the class probabilities for the given features using the fitted model.

        Parameters:
        ----------
        features : ndarray
            The input features to be transformed and used for predicting probabilities.

        Returns:
        -------
        ndarray
            The predicted class probabilities.
        """
        transformed_features = self.kernel.transform(features)
        return self.model.predict_proba(transformed_features)