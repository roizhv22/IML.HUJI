from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.ndarray((len(self.classes_),))
        self.mu_ = np.ndarray((len(self.classes_), len(X[0])))
        self.vars_ = np.ndarray((len(self.classes_), len(X[0])))
        m = len(X)

        for j in range(len(X[0])):
            for k in range(len(self.classes_)):
                n_k = (y == self.classes_[k]).sum()
                self.pi_[k] = (n_k / m)
                val = 0
                for i in range(m):
                    if y[i] == self.classes_[k]:
                        val += X[i][j]
                self.mu_[k][j] = (1 / n_k) * val

            for k in range(len(self.classes_)):
                n_k = (y == self.classes_[k]).sum()
                val = 0
                for i in range(m):
                    if y[i] == self.classes_[k]:
                        val += (X[i][j] - self.mu_[k][j]) * (X[i][j] -
                                                             self.mu_[k][j])
                self.vars_[k][j] = (1 / (n_k-1)) * val  # sigma^2

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """

        arg_max_ind = np.argmax(self.likelihood(X), axis=1)
        # get the argmax in each column, which is the argmax in each smaple,
        # as the prediction is based on MLE principle.
        res = []
        for i in range(len(X)):
            res.append(self.classes_[arg_max_ind[i]]) # get arg max by class,
            # via the likelihood matrix
        return np.array(res)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        res = np.ndarray((len(X), len(self.classes_)))

        for k in range(len(self.classes_)):
            cov = np.diag(self.vars_[k])
            const = 1 / np.sqrt(
                np.power(2 * np.pi, len(X)) * np.linalg.det(cov))
            vec = np.diag((X - self.mu_[k]) @ np.linalg.inv(cov) @
                          np.transpose((X - self.mu_[k])))
            res[:, k] = self.pi_[k] * const * np.exp((-1 / 2) * vec)

        return res

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        from ...metrics import misclassification_error
        return misclassification_error(y, self._predict(X))

