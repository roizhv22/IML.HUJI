from __future__ import annotations
from typing import NoReturn

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import LinearSVC

from IMLearn.base import BaseEstimator
import numpy as np
import pandas as pd


from IMLearn.metrics import misclassification_error


class AgodaCancellationEstimator(BaseEstimator):
    """
    An estimator for solving the Agoda Cancellation challenge
    """

    models = []

    def __init__(self) -> AgodaCancellationEstimator:
        """
        Instantiate an estimator for solving the Agoda Cancellation challenge

        Parameters
        ----------


        Attributes
        ----------

        """
        super().__init__()

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit an estimator for given samples

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to

        Notes
        -----

        """
        # Split the Data into Training and Testing sets with test size as #30%
        # Data scaa scaling
        std_scaler = StandardScaler()
        std_scaler.fit(X)
        self.models.append(std_scaler)
        X_train_std = std_scaler.transform(X)
        # X_test_std = std_scaler.transform(y)

        mm_scaler = MinMaxScaler()
        mm_scaler.fit(X)
        self.models.append(std_scaler)
        X_train_mm = mm_scaler.transform(X)
        # X_test_mm = mm_scaler.transform(y)

        # Logistic Regression
        logreg = LogisticRegression(max_iter=500).fit(X_train_mm, y)
        self.models.append(logreg)

        # Linear SVC
        svc = LinearSVC().fit(X_train_mm, y)
        self.models.append(svc)

        # SGD Classifier
        sgd = SGDClassifier(alpha=0.1).fit(X_train_std, y)
        self.models.append(sgd)

        # KNN
        neighbors_settings = range(1, 6)
        for n_neighbors in neighbors_settings:
            knn = KNeighborsClassifier(n_neighbors=n_neighbors)
            knn.fit(X, y)
            self.models.append(knn)


    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given ####samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pred = np.ndarray((len(X),))
        for model in self.models:
            pred += model.predict(X)  # TODD - check

        return np.ndarray([pred[i] // len(self.models) for i in range(len(X))])
        # treshold = 0.5


    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under loss function
        """
        return misclassification_error(y, self._predict(X))
