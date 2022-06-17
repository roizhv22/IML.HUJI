import numpy as np
import pandas as pd
from typing import Tuple, List, Callable, Type

from IMLearn import BaseModule
from IMLearn.desent_methods import GradientDescent, FixedLR, ExponentialLR
from IMLearn.desent_methods.modules import L1, L2
from IMLearn.learners.classifiers.logistic_regression import LogisticRegression
from IMLearn.utils import split_train_test

import plotly.graph_objects as go


def plot_descent_path(module: Type[BaseModule],
                      descent_path: np.ndarray,
                      title: str = "",
                      xrange=(-1.5, 1.5),
                      yrange=(-1.5, 1.5)) -> go.Figure:
    """
    Plot the descent path of the gradient descent algorithm

    Parameters:
    -----------
    module: Type[BaseModule]
        Module type for which descent path is plotted

    descent_path: np.ndarray of shape (n_iterations, 2)
        Set of locations if 2D parameter space being the regularization path

    title: str, default=""
        Setting details to add to plot title

    xrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    yrange: Tuple[float, float], default=(-1.5, 1.5)
        Plot's x-axis range

    Return:
    -------
    fig: go.Figure
        Plotly figure showing module's value in a grid of [xrange]x[yrange] over which regularization path is shown

    Example:
    --------
    fig = plot_descent_path(IMLearn.desent_methods.modules.L1, np.ndarray([[1,1],[0,0]]))
    fig.show()
    """
    def predict_(w):
        return np.array([module(weights=wi).compute_output() for wi in w])

    from utils import decision_surface
    return go.Figure([decision_surface(predict_, xrange=xrange, yrange=yrange, density=70, showscale=False),
                      go.Scatter(x=descent_path[:, 0], y=descent_path[:, 1], mode="markers+lines", marker_color="black")],
                     layout=go.Layout(xaxis=dict(range=xrange),
                                      yaxis=dict(range=yrange),
                                      title=f"GD Descent Path {title}"))


def get_gd_state_recorder_callback() -> Tuple[Callable[[], None], List[np.ndarray], List[np.ndarray]]:
    """
    Callback generator for the GradientDescent class, recording the objective's value and parameters at each iteration

    Return:
    -------
    callback: Callable[[], None]
        Callback function to be passed to the GradientDescent class, recoding the objective's value and parameters
        at each iteration of the algorithm

    values: List[np.ndarray]
        Recorded objective values

    weights: List[np.ndarray]
        Recorded parameters
    """

    res_weights = []
    res_values = []
    def callback(output, weights, grad, t, eta):
        res_weights.append(weights)
        res_values.append(output)

    return callback, res_values, res_weights



def compare_fixed_learning_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                 etas: Tuple[float] = (1, .1, .01, .001)):


    for model in [L1, L2]:
        cb_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(FixedLR(.01), callback=cb_l1)
        gd_l1.fit(model(init), X=None, y=None)
        f1 = plot_descent_path(model, np.array(weights_l1), f"decent path for "
                                    f"{model.__name__} module with eta={0.01}")
        f1.write_image(f"ex6/Q1/GD_{model.__name__}.jpeg")

        for eta in etas:
            cb_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
            gd_l1 = GradientDescent(FixedLR(eta), callback=cb_l1)
            gd_l1.fit(model(init), X=None, y=None)
            cov_plot = go.Figure(data=
                [go.Scatter(x=[i for i in range(len(values_l1))], y=values_l1,
                    mode="markers+lines",
                            name="convergence rate")])
            cov_plot.update_layout(title=f"EX3 with eta={eta}",
                                   xaxis_title="GD iteration"
                                   , yaxis_title="norm value")
            cov_plot.write_image(f"ex6/Q3/cov_{eta}_{model.__name__}.jpeg")
            print(f"min for eta {eta} in module {model.__name__} is "
                  f"{np.min(values_l1)}")



def compare_exponential_decay_rates(init: np.ndarray = np.array([np.sqrt(2), np.e / 3]),
                                    eta: float = .1,
                                    gammas: Tuple[float] = (.9, .95, .99, 1)):
    # Optimize the L1 objective using different decay-rate values
    # of the exponentially decaying learning rate
    values_per_gamma = []
    min_of_l1 = np.inf
    min_gamma = 0
    for gamma in gammas:
        cb_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=ExponentialLR(eta, gamma),
                                callback=cb_l1)
        gd_l1.fit(L1(init), X=None, y=None)
        values_per_gamma.append(values_l1)
        if min_of_l1 > np.min(values_l1):
            min_of_l1 = np.min(values_l1)
            min_gamma = gamma
    iters = len(values_per_gamma[0])
    conv_fig = go.Figure(data=[go.Scatter(x=[i for i in range(iters)], y=value,
                         name=f"CR with gamma = {gammas[j]}",
                         mode="markers+lines")
                         for j, value in enumerate(values_per_gamma)])
    conv_fig.update_layout(title="EX5",xaxis_title="GD iteration",
                         yaxis_title="norm value")
    conv_fig.write_image("ex6/Q5/cov_rate.jpeg")
    print(f"min of l_1 for all decay rates is {min_of_l1} "
          f"for gamma {min_gamma}")

    # Plot descent path for gamma=0.95
    for module in [L1, L2]:
        cb_l1, values_l1, weights_l1 = get_gd_state_recorder_callback()
        gd_l1 = GradientDescent(learning_rate=ExponentialLR(eta, 0.95),
                                callback=cb_l1)
        gd_l1.fit(module(init), X=None, y=None)
        f1 = plot_descent_path(module, np.array(weights_l1), f"decent path for "
                                    f"{module.__name__} module with eta={0.1}"
                                                             f", gamma = 0.95")
        f1.write_image(f"ex6/Q7/GD_{module.__name__}_gamma_0.95.jpeg")


def load_data(path: str = "../datasets/SAheart.data", train_portion: float = .8) -> \
        Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Load South-Africa Heart Disease dataset and randomly split into a train- and test portion

    Parameters:
    -----------
    path: str, default= "../datasets/SAheart.data"
        Path to dataset

    train_portion: float, default=0.8
        Portion of dataset to use as a training set

    Return:
    -------
    train_X : DataFrame of shape (ceil(train_proportion * n_samples), n_features)
        Design matrix of train set

    train_y : Series of shape (ceil(train_proportion * n_samples), )
        Responses of training samples

    test_X : DataFrame of shape (floor((1-train_proportion) * n_samples), n_features)
        Design matrix of test set

    test_y : Series of shape (floor((1-train_proportion) * n_samples), )
        Responses of test samples
    """
    df = pd.read_csv(path)
    df.famhist = (df.famhist == 'Present').astype(int)
    return split_train_test(df.drop(['chd', 'row.names'], axis=1), df.chd, train_portion)


def fit_logistic_regression():
    # Load and split SA Heard Disease dataset
    X_train, y_train, X_test, y_test = load_data()

    # Plotting convergence rate of logistic regression over SA heart disease data
    raise NotImplementedError()

    # Fitting l1- and l2-regularized logistic regression models, using cross-validation to specify values
    # of regularization parameter
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    # compare_fixed_learning_rates()
    compare_exponential_decay_rates()
    # fit_logistic_regression()
