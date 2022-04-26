from typing import NoReturn
from ...base import BaseEstimator
import numpy as np
from numpy.linalg import det, inv


class LDA(BaseEstimator):
    """
    Linear Discriminant Analysis (LDA) classifier

    Attributes
    ----------
    self.classes_ : np.ndarray of shape (n_classes,)
        The different labels classes. To be set in `LDA.fit`

    self.mu_ : np.ndarray of shape (n_classes,n_features)
        The estimated features means for each class. To be set in `LDA.fit`

    self.cov_ : np.ndarray of shape (n_features,n_features)
        The estimated features covariance. To be set in `LDA.fit`

    self._cov_inv : np.ndarray of shape (n_features,n_features)
        The inverse of the estimated features covariance. To be set in `LDA.fit`

    self.pi_: np.ndarray of shape (n_classes)
        The estimated class probabilities. To be set in `LDA.fit`
    """

    def __init__(self):
        """
        Instantiate an LDA classifier
        """
        super().__init__()
        self.classes_, self.mu_, self.cov_, self._cov_inv, self.pi_ = None,\
                                                    None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits an LDA model.
        Estimates gaussian for each label class - Different mean vector,
        same covariance matrix with dependent features.

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
        self.mu_map_ = {}
        m = len(y)
        # find pi + mu_k
        for i in range(len(self.classes_)):
            n_k = (y == self.classes_[i]).sum()
            self.pi_[i] = n_k / m
            mu_vec = np.zeros((len(X[0]),))
            for j in range(m):
                if y[j] == self.classes_[i]:
                    mu_vec += X[j]
            self.mu_[i] = (1 / n_k) * mu_vec
            self.mu_map_[self.classes_[i]] = self.mu_[i]

        # find var
        self.cov_ = np.zeros((len(X[0]), len(X[0])))
        for i in range(m):
            vec = np.matrix((X[i] - self.mu_map_[y[i]]))
            self.cov_ += (vec.T @ vec)
        self.cov_ = self.cov_ * (1/m)
        self._cov_inv = np.linalg.inv(self.cov_)




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
        a_k = np.ndarray((len(self.classes_),len(self.classes_)-1))
        b_k = np.ndarray((len(self.classes_),1))
        res = np.ndarray((len(X),))
        for i in range(len(self.classes_)):
            a_k[i] = self._cov_inv @ self.mu_[i]
            b_k[i] = np.log(self.pi_[i]) - (1 / 2) * (
                        self.mu_[i] @ self._cov_inv @
                        self.mu_[i])

        for i in range(len(X)):
            vals = np.ndarray((len(self.classes_),))
            for j in range(len(self.classes_)):
                vals[j] = (np.transpose(a_k[j]) @ X[i]) + b_k[j]
            res[i] = self.classes_[np.argmax(vals)]

        return res

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
            raise ValueError("Estimator must first be fitted before "
                             "calling `likelihood` function")

        res = np.zeros((len(X), len(self.classes_)))
        const = 1 / np.sqrt(
            np.power(2 * np.pi, len(X)) * np.linalg.det(self.cov_))


        for k in range(len(self.classes_)):
            vec = np.diag((X - self.mu_[k]) @ self._cov_inv @ np.transpose((X -
                                                            self.mu_[k])))
            res[:, k] = self.pi_[k] * const * np.exp((-1/2) * vec)

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
        pred = self._predict(X)
        return misclassification_error(y, pred)
