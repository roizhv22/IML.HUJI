import numpy as np

from IMLearn.learners.classifiers import Perceptron, LDA, GaussianNaiveBayes
from typing import Tuple
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from math import atan2, pi


def load_dataset(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset for comparing the Gaussian Naive Bayes and LDA classifiers. File is assumed to be an
    ndarray of shape (n_samples, 3) where the first 2 columns represent features and the third column the class

    Parameters
    ----------
    filename: str
        Path to .npy data file

    Returns
    -------
    X: ndarray of shape (n_samples, 2)
        Design matrix to be used

    y: ndarray of shape (n_samples,)
        Class vector specifying for each sample its class

    """
    data = np.load(filename)
    return data[:, :2], data[:, 2].astype(int)


def run_perceptron():
    """
    Fit and plot fit progression of the Perceptron algorithm over both the linearly separable and inseparable datasets

    Create a line plot that shows the perceptron algorithm's training loss values (y-axis)
    as a function of the training iterations (x-axis).
    """
    for n, f in [("Linearly Separable", "linearly_separable.npy"),
                 ("Linearly Inseparable", "linearly_inseparable.npy")]:
        # Load dataset
        X, y = load_dataset(f"C:/Users/roizh/Desktop/IML.HUJI/datasets/{f}")

        # Fit Perceptron and record loss in each fit iteration
        losses = []
        cb = lambda precp, sample, response: losses.append(
            precp.loss(sample, response))
        percep = Perceptron(callback=cb)
        percep.fit(X, y)
        # Plot figure of loss as function of fitting iteration
        plot = go.Figure(
            go.Scatter(x=[i for i in range(len(losses))], y=losses,
                       mode="markers"),
            layout=go.Layout(title=f"{n} Preceptron's loss to iteration plot",
                             xaxis_title="Iterations",
                             yaxis_title="Model's loss"))
        plot.write_image(
            f"C:/Users/roizh/Desktop/IML.HUJI/"
            f"exercises/ex3/plots/{n} Preceptron's results.png")


def get_ellipse(mu: np.ndarray, cov: np.ndarray):
    """
    Draw an ellipse centered at given location and according to specified covariance matrix

    Parameters
    ----------
    mu : ndarray of shape (2,)
        Center of ellipse

    cov: ndarray of shape (2,2)
        Covariance of Gaussian

    Returns
    -------
        scatter: A plotly trace object of the ellipse
    """
    l1, l2 = tuple(np.linalg.eigvalsh(cov)[::-1])
    theta = atan2(l1 - cov[0, 0], cov[0, 1]) if cov[0, 1] != 0 else (
        np.pi / 2 if cov[0, 0] < cov[1, 1] else 0)
    t = np.linspace(0, 2 * pi, 100)
    xs = (l1 * np.cos(theta) * np.cos(t)) - (l2 * np.sin(theta) * np.sin(t))
    ys = (l1 * np.sin(theta) * np.cos(t)) + (l2 * np.cos(theta) * np.sin(t))

    return go.Scatter(x=mu[0] + xs, y=mu[1] + ys, mode="lines",
                      marker_color="black")


def compare_gaussian_classifiers():
    """
    Fit both Gaussian Naive Bayes and LDA classifiers on both
    gaussians1 and gaussians2 datasets
    """
    for f in ["gaussian1.npy", "gaussian2.npy"]:
        # Load dataset
        X, y = load_dataset(f"../datasets/{f}")
        # Fit models and predict over training set
        naive = GaussianNaiveBayes()
        lda = LDA()
        lda.fit(X, y)
        naive.fit(X, y)
        naive_pred = naive.predict(X)
        lda_pred = lda.predict(X)

        # Plot a figure with two subplots, showing the Gaussian Naive
        # Bayes predictions on the left and LDA predictions
        # on the right. Plot title should specify dataset used and
        # subplot titles should specify algorithm and accuracy
        # Create subplots
        from IMLearn.metrics import accuracy
        fig = make_subplots(1, 2,
                            subplot_titles=[f"Gaussian Naive Bayes, accuracy "
                                            f"{accuracy(y, naive_pred)}",
                                            f"LDA, accuracy "
                                            f"{accuracy(y, lda_pred)}"])
        fig.update_layout(title={"text": f})
        # naive
        fig.add_scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                        marker=dict(color=naive_pred, symbol=y),
                        text=f"Gaussian Naive Bayes, accuracy "
                             f"{accuracy(y, naive_pred)}", row=1,
                        col=1)

        # LDA
        fig.add_scatter(x=X[:, 0], y=X[:, 1], mode="markers",
                        marker=dict(color=lda_pred, symbol=y), xaxis="x",
                        row=1,
                        col=2)
        fig.update_xaxes(title_text="Feature 1", row=1, col=1)
        fig.update_xaxes(title_text="Feature 1", row=1, col=2)
        fig.update_yaxes(title_text="Feature 2", row=1, col=1)
        fig.update_yaxes(title_text="Feature 2", row=1, col=2)

        # Add `X` dots specifying fitted Gaussians' means
        fig.add_scatter(x=lda.mu_[:, 0], y=lda.mu_[:, 1], mode="markers",
                        marker=dict(color="black", symbol="x"),
                        row=1, col=1)
        fig.add_scatter(x=naive.mu_[:, 0], y=naive.mu_[:, 1], mode="markers",
                        marker=dict(color="black", symbol="x"),
                        row=1, col=2)

        # Add ellipses depicting the covariances of the fitted Gaussians
        fig.add_trace(get_ellipse(lda.mu_[0], lda.cov_), col=2, row=1)
        fig.add_trace(get_ellipse(lda.mu_[1], lda.cov_), col=2, row=1)
        fig.add_trace(get_ellipse(lda.mu_[2], lda.cov_), col=2, row=1)
        fig.add_trace(get_ellipse(naive.mu_[0], np.diag(naive.vars_[0])),
                      col=1, row=1)
        fig.add_trace(get_ellipse(naive.mu_[1], np.diag(naive.vars_[1])),
                      col=1, row=1)
        fig.add_trace(get_ellipse(naive.mu_[2], np.diag(naive.vars_[2])),
                      col=1, row=1)

        fig.show()


def quiz():
    q_1_X = [[i] for i in range(8)]
    q_1_y = [0, 0, 1, 1, 1, 1, 2, 2]
    naive = GaussianNaiveBayes()
    naive.fit(q_1_X, q_1_y)
    print(f"Q1 pi{naive.pi_[0]} mu1{naive.mu_[1]}")

    q_2_X = [[1, 1], [1, 2], [2, 3], [2, 4], [3, 3], [3, 4]]
    q_2_y = [0, 0, 1, 1, 1, 1]
    naive.fit(q_2_X, q_2_y)
    print(naive.vars_)







if __name__ == '__main__':
    np.random.seed(0)
    run_perceptron()
    compare_gaussian_classifiers()
    # quiz()
