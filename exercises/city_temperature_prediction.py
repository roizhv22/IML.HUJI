import datetime

import plotly
from pygments.lexers import go

import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from IMLearn.metrics.loss_functions import mean_square_error

pio.templates.default = "simple_white"


def load_data(filename: str):
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = pd.read_csv(filename, parse_dates=['Date'])
    df = df[df.Temp > -72]  # filter out invalid samples with -72.xxx
    df['DayOfYear'] = df['Date'].apply(lambda x: x.day_of_year)
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    X = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    Israel_X = X.loc[X['Country'] == 'Israel']
    df_for_1 = Israel_X[['DayOfYear', 'Temp']].reset_index(drop=True)
    fig1 = px.scatter(df_for_1, x='DayOfYear', y='Temp',
                      color=Israel_X['Year'].astype(str),
                      title="Q2.1 - Temp to DayOfYear Israel")
    fig1.write_image("ex2/plots/poly Q2.1.png")
    df_for_2 = Israel_X.groupby("Month").agg(np.std)

    df_for_2["Temp standard deviation"] = df_for_2['Temp']
    fig2 = px.bar(df_for_2, y="Temp standard deviation", text_auto=True,
                  color=Israel_X['Month'].unique().astype(str))
    fig2.write_image("ex2/plots/poly Q2.2.png")

    # Question 3 - Exploring differences between countries
    Q3_df_avg = X.groupby(["Country", "Month"]).agg(np.average)
    Q3_df_std = X.groupby(["Country", "Month"]).agg(np.std)

    fig3 = go.Figure(data=[go.Scatter(x=[i for i in range(1, 13)],
                                      y=Q3_df_avg.loc[x]['Temp'],
                                      error_y=
                                      plotly.graph_objs.scatter.
                                      ErrorY(array=Q3_df_std.loc[x]['Temp']),
                                      name=x)
                           for x in X['Country'].unique()],
                     layout=go.Layout(title="Polyfit Q3",
                                      xaxis_title="Month",
                                      yaxis_title="Avg temp"))
    fig3.write_image("ex2/plots/poly_Q3.png")

    # Question 4 - Fitting model for different values of `k`
    train_Israel = Israel_X.sample(frac=0.75)
    test_Israel = Israel_X.drop(train_Israel.index)
    loss_per_deg = []
    for i in range(1, 11):
        polyfit_model = PolynomialFitting(i)
        polyfit_model.fit(train_Israel['DayOfYear'], train_Israel['Temp'])
        val = round(polyfit_model.loss(test_Israel['DayOfYear'],
                                       test_Israel['Temp']), 2)
        loss_per_deg.append(val)
        print(f"(k={i}, {val})")

    degs = [str(i) for i in range(1, 11)]
    fig4 = px.bar(x=[i for i in range(1, 11)], y=loss_per_deg, text_auto=True,
                  color=degs, title="Q4 polyfit - Model MSE pre degree",
                  labels={"x": "Polynomial degree", "y": "MSE of model"})
    fig4.update_layout()
    fig4.write_image("ex2/plots/poly_Q4.png")

    # Question 5 - Evaluating fitted model on different countries
    isra_model = PolynomialFitting(6)
    isra_model.fit(Israel_X['DayOfYear'], Israel_X['Temp'])
    errors = []
    countries = ['South Africa', 'The Netherlands', 'Jordan']
    for country in countries:
        cur_c = X.loc[X['Country'] == country]
        errors.append(isra_model.loss(cur_c['DayOfYear'], cur_c['Temp']))
    fig5 = px.bar(x=countries, y=errors, text_auto=True, color=countries,
                  labels={"x": "Country", "y": "Model MSE"},
                  title="Q5 polyfit - Israel_Model MSE for other countries, k=5")
    fig5.write_image("ex2/plots/poly_Q5.png")

