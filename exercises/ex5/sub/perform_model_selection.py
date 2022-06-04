from __future__ import annotations

import numpy
import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """
    # Question 1 - Generate dataset for model
    # f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    axis = np.random.uniform(-1.2, 2, n_samples)

    def poly(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    y = np.array([poly(x) for x in axis])

    # set datasets
    train_X, train_y, test_X, test_y = split_train_test(axis, y, 0.6666667)
    train_y += np.random.normal(0, noise, len(train_y))
    test_y += np.random.normal(0, noise, len(test_y))

    x1 = numpy.take(axis, train_X.index)
    x2 = numpy.take(axis, test_X.index)

    train_y = train_y.to_numpy()
    train_X = train_X[0].to_numpy()
    test_X = test_X[0].to_numpy()
    test_y = test_y.to_numpy()

    fig1 = go.Figure(
        [go.Scatter(x=axis, y=y, mode="markers", name="Noiseless"),
         go.Scatter(x=x1, y=train_y, mode="markers",
                    name="Train"),
         go.Scatter(x=x2, y=test_y, mode="markers",
                    name="Test")]
        , layout=go.Layout(title=f"Q1,noise = {noise}, n={n_samples}", xaxis_title="x", yaxis_title="y"))
    fig1.write_image(f"ex5/Q1 noise = {noise}, n={n_samples}.jpeg")
    # fig1.show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train, validate = [], []

    for k in range(11):
        est = PolynomialFitting(k)
        t, val = cross_validate(est, train_X,
                                train_y,
                                mean_square_error)
        train.append(t)
        validate.append(val)
    fig2 = go.Figure(
        [go.Scatter(x=list(range(11)), y=train, mode="markers+lines",
                    name="Train score"),
         go.Scatter(x=list(range(11)), y=validate, mode="markers+lines",
                    name="Validation score")]
        , layout=go.Layout(title=f"Q2, noise = {noise}, n={n_samples}", xaxis_title="K", yaxis_title="Score"))
    fig2.write_image(f"ex5/Q2 noise = {noise}, n={n_samples}.jpeg")
    # fig2.show()

    # Q3
    k_star = np.argmin(validate)
    poly = PolynomialFitting(k_star)
    poly.fit(train_X, train_y)

    print(f"K^* = {k_star}, error is {poly.loss(test_X, test_y)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True)
    test_X, test_y, train_X, train_y = X[:n_samples], y[:n_samples], \
                                       X[n_samples:], y[n_samples:]
    lam_vals = np.linspace(0, 0.5, 500)

    # Question 7 - Perform CV for different values of the
    # regularization parameter for Ridge and Lasso regressions
    ridge_train, ridge_validate = [], []
    lasso_train, lasso_validate = [], []
    for lam in lam_vals:
        ridge = RidgeRegression(lam)
        lasso = Lasso(alpha=lam)
        ridge_t, ridge_v = cross_validate(ridge, train_X, train_y,
                                          mean_square_error)
        lasso_t, lasso_v = cross_validate(lasso, train_X, train_y,
                                          mean_square_error)
        ridge_train.append(ridge_t)
        ridge_validate.append(ridge_v)
        lasso_train.append(lasso_t)
        lasso_validate.append(lasso_v)

    fig1 = go.Figure(
        [go.Scatter(x=lam_vals, y=ridge_train, mode="markers+lines",
                    name="Train score"),
         go.Scatter(x=lam_vals, y=ridge_validate, mode="markers+lines",
                    name="Validation score")]
        , layout=go.Layout(title="Ridge", xaxis_title="Lambda",
                           yaxis_title="Error"))
    fig1.write_image(f"ex5/Q7 Ridge.jpeg")
    best_for_ridge = lam_vals[np.argmin(ridge_validate)]
    # fig1.show()
    fig2 = go.Figure(
        [go.Scatter(x=lam_vals, y=lasso_train, mode="markers+lines",
                    name="Train score"),
         go.Scatter(x=lam_vals, y=lasso_validate, mode="markers+lines",
                    name="Validation score")]
        , layout=go.Layout(title="Lasso", xaxis_title="Lambda",
                           yaxis_title="Error"))
    fig2.write_image(f"ex5/Q7 lasso.jpeg")
    # fig2.show()
    best_for_lasso = lam_vals[np.argmin(lasso_validate)]
    ridge = RidgeRegression(best_for_ridge)
    lasso = Lasso(alpha=best_for_lasso)
    lr = LinearRegression()
    ridge.fit(train_X, train_y)
    lasso.fit(train_X, train_y)
    lr.fit(train_X, train_y)
    print(f"Ridge achieved {ridge.loss(test_X,test_y)} with lambda of "
          f"{best_for_ridge}")
    print(f"Lasso achieved {mean_square_error(test_y, lasso.predict(test_X))}"
          f" with lambda of {best_for_lasso}")
    print(f"Basic linear regression achieved {lr.loss(test_X, test_y)}")



if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
