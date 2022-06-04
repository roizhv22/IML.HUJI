from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """

    split_X = np.array_split(X, cv)
    split_y = np.array_split(y, cv)
    validation = 0
    train_err = 0

    for fold in range(cv):
        validate_X, validate_y = split_X[fold], split_y[fold]
        train_X, train_Y = np.concatenate(
            (split_X[:fold] + split_X[fold + 1:]), axis=0), \
                           np.concatenate(
                               (split_y[:fold] + split_y[fold + 1:]), axis=0)
        estimator.fit(train_X, train_Y)
        validation += scoring(validate_y, estimator.predict(validate_X))
        train_err += scoring(train_Y, estimator.predict(train_X))
    return train_err / cv, validation / cv
