
from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = pd.read_csv(filename)
    df = df.dropna()
    df = df.loc[~(df == 0).all(axis=1)]  # remove lines of 0s

    df["is_renovated"] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)
    df['basement_size'] = df['sqft_basement'].apply(lambda x: x / 1000)
    df['age'] = df['yr_built'].apply(lambda x: 1 / (2022 - x))
    # there are no houses that built in 2022, so we will not divided in 0

    response = df['price']
    sample_mat = df[['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot',
                     'floors', 'waterfront', 'view', 'condition', 'grade',
                     'sqft_above', 'is_renovated', 'basement_size', 'age']]

    return sample_mat, response


def _calculate_pearson_correlation(x, y):
    """
    Calculate the Pearson correlation as described in the exercise pdf.
    """
    to_mean = (x - np.mean(x)) * (y - np.mean(y))
    return np.mean(to_mean) / (np.std(x) * np.std(y))


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        pearson_corr = _calculate_pearson_correlation(X[feature].to_numpy(),
                                                      y.to_numpy())
        fig = go.Figure(go.Scatter(x=X[feature], y=y, mode='markers', ),
                        layout=go.Layout(
                            title=f"{feature} -"
                                  f" Pearson Correlation is {pearson_corr}",
                            xaxis_title=f"{feature} ",
                            yaxis_title="Response data"))
        fig.write_image(output_path + f"{feature}.png")


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data(
        "C:/Users/roizh/Desktop/IML.HUJI/datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y, "ex2/plots/")  # output to be set.

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y, 0.75)

    # Question 4 - Fit model over increasing percentages of the overall
    # training data For every percentage p in 10%, 11%, ..., 100%, repeat
    # the following 10 times: 1) Sample p% of the overall training data 2)
    # Fit linear model (including intercept) over sampled set 3) Test fitted
    # model over test set 4) Store average and variance of loss over test
    # set Then plot average loss as function of training size with error
    # ribbon of size (mean-2*std, mean+2*std)

    percentage = []
    loss_per_p = []
    upper = []
    lower = []
    for p in range(10, 101):
        percentage.append(p)
        samples = []
        for i in range(10):
            cur_X = train_X.sample(frac=p/100)
            cur_y = train_y.reindex_like(cur_X)
            model = LinearRegression()
            model.fit(cur_X.to_numpy(), cur_y.to_numpy())
            samples.append(model.loss(test_X.to_numpy(), test_y.to_numpy()))

        mean = np.mean(samples)
        std = np.std(samples)
        loss_per_p.append(mean)
        upper.append(mean + 2*std)
        lower.append(mean - 2*std)


    fig = go.Figure(
        [go.Scatter(x=percentage, y=loss_per_p, mode='markers+lines',name="RSS"),
         go.Scatter(x=percentage, y=lower, fill=None,
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False),
         go.Scatter(x=percentage, y=upper, fill='tonexty',
                    mode="lines", line=dict(color="lightgrey"),
                    showlegend=False)],
        layout=go.Layout(
            title="RSS per train data percentage - Linear regression",
            xaxis_title="Train data percentage",
            yaxis_title="RSS"))
    fig.write_image("ex2/plots/Q4.png")






