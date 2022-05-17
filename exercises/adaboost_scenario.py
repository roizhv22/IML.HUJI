from functools import cached_property

import numpy as np
from typing import Tuple

import tqdm

from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)
    # Question 1: Train- and test errors of AdaBoost in noiseless case
    err_on_test = []
    err_on_smaple = []
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    x = [i for i in range(1, 251)]
    for i in tqdm.tqdm(range(250)):
        err_on_test.append(adaboost.partial_loss(test_X, test_y, i))
        err_on_smaple.append(adaboost.partial_loss(train_X, train_y, i))

    f1 = go.Figure(data=[go.Scatter(x=x, y=err_on_test, mode="markers+lines",
                                    name="error on test"),
                         go.Scatter(x=x, y=err_on_smaple, mode="markers+lines",
                                    name="error on train")],
                   layout=go.Layout(title="Q1 error of fitted AdaBoost model",
                                    xaxis_title="Number of models in use",
                                    yaxis_title="Model's error"))
    f1.write_image(f"ex4/Q1 noise {noise}.jpeg")
    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])

    fig2 = make_subplots(rows=2, cols=2,
                         subplot_titles=[f"{m} iterations" for m in T],
                         horizontal_spacing=0.01, vertical_spacing=.1)
    for i, m in enumerate(T):
        fig2.add_traces([decision_surface(
            lambda X: adaboost.partial_predict(X, m), lims[0], lims[1],
            showscale=False),
                         go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                    mode="markers",
                                    showlegend=False,
                                    marker=dict(color=test_y,
                                                colorscale=[custom[0],
                                                            custom[-1]],
                                                line=dict(color="black",
                                                          width=1)))],
                        rows=(i // 2) + 1, cols=(i % 2) + 1)
    fig2.update_layout(title="Q2 - decision surface as size of committee")
    fig2.write_image(f"ex4/Q2 noise {noise}.jpeg")

    # Question 3: Decision surface of best performing ensemble
    best_size = np.argmin(err_on_test)
    fig3 = go.Figure(data=[decision_surface(lambda X:
                                            adaboost.partial_predict(X,best_size),
                                            lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color=test_y,
                                                  colorscale=[custom[0],
                                                              custom[-1]],
                                                  line=dict(color="black",
                                                            width=1)))],
                     layout=go.Layout(
                         title=f"Q3 - decision surface of {best_size} "
                               f"members, with accuracy of {1 - err_on_test[best_size]}"))
    fig3.write_image(f"ex4/Q3 noise {noise}.jpeg")

    # Question 4: Decision surface with weighted samples
    s = adaboost.D_/np.max(adaboost.D_) * 5.0
    fig4 = go.Figure(data=[decision_surface(adaboost.predict,
                                            lims[0], lims[1],
                                            showscale=False),
                           go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                                      mode="markers",
                                      showlegend=False,
                                      marker=dict(color=train_y, size=s,
                                                  colorscale=[custom[0],
                                                              custom[-1]],
                                                  line=dict(color="black",
                                                            width=1)))],
                     layout=go.Layout(
                         title=f"Q4 - Weights samples, noise {noise}"))
    fig4.write_image(f"ex4/Q4 noise {noise}.jpeg")


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
