import math

import tqdm
from sklearn.base import BaseEstimator

from challenge.agoda_cancellation_estimator import AgodaCancellationEstimator
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd

import time


def load_data(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    # features = full_data[["h_booking_id",
    #                       "hotel_id",
    #                       "accommadation_type_name",
    #                       "hotel_star_rating",
    #                       "customer_nationality"]]
    full_data['booking_datetime'] = pd.to_datetime(
        full_data['booking_datetime'])
    full_data['booking_datetime_year'] = full_data['booking_datetime'].dt.year
    full_data['booking_datetime_month'] = full_data[
        'booking_datetime'].dt.month
    full_data['booking_datetime_week'] = full_data['booking_datetime'].dt.week
    full_data['booking_datetime_day'] = full_data['booking_datetime'].dt.day
    full_data['booking_datetime_hour'] = full_data['booking_datetime'].dt.hour
    full_data['booking_datetime_minute'] = full_data[
        'booking_datetime'].dt.minute
    full_data['booking_datetime_day_of_week'] = full_data[
        'booking_datetime'].dt.dayofweek
    full_data = full_data.drop(["booking_datetime"], axis=1)

    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkin_date_year'] = full_data['checkin_date'].dt.year
    full_data['checkin_date_month'] = full_data['checkin_date'].dt.month
    full_data['checkin_date_week'] = full_data['checkin_date'].dt.week
    full_data['checkin_date_day'] = full_data['checkin_date'].dt.day
    full_data['checkin_date_hour'] = full_data['checkin_date'].dt.hour
    full_data['checkin_date_minute'] = full_data['checkin_date'].dt.minute
    full_data['checkin_date_day_of_week'] = full_data[
        'checkin_date'].dt.dayofweek
    full_data = full_data.drop(["checkin_date"], axis=1)

    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])
    full_data['checkout_date_year'] = full_data['checkout_date'].dt.year
    full_data['checkout_date_month'] = full_data['checkout_date'].dt.month
    full_data['checkout_date_week'] = full_data['checkout_date'].dt.week
    full_data['checkout_date_day'] = full_data['checkout_date'].dt.day
    full_data['checkout_date_hour'] = full_data['checkout_date'].dt.hour
    full_data['checkout_date_minute'] = full_data['checkout_date'].dt.minute
    full_data['checkout_date_day_of_week'] = full_data[
        'checkout_date'].dt.dayofweek
    full_data = full_data.drop(["checkout_date"], axis=1)

    full_data['hotel_live_date'] = pd.to_datetime(full_data['hotel_live_date'])
    full_data['hotel_live_date_year'] = full_data['hotel_live_date'].dt.year
    full_data['hotel_live_date_month'] = full_data['hotel_live_date'].dt.month
    full_data['hotel_live_date_week'] = full_data['hotel_live_date'].dt.week
    full_data['hotel_live_date_day'] = full_data['hotel_live_date'].dt.day
    full_data['hotel_live_date_hour'] = full_data['hotel_live_date'].dt.hour
    full_data['hotel_live_date_minute'] = full_data[
        'hotel_live_date'].dt.minute
    full_data['hotel_live_date_day_of_week'] = full_data[
        'hotel_live_date'].dt.dayofweek
    full_data = full_data.drop(["hotel_live_date"], axis=1)

    # add dummies to parse the cancelation policy
    interesting_policies = ['1D1N_1N', '1D100P', '3D1N_1N', '3D1N_100P',
                            '3D100P', '365D100P_100P',
                            '2D100P', '3D100P_100P', '1D100P_100P',
                            '7D100P_100P', '7D1N_100P',
                            '0D0N', '7D100P', '2D1N_1N', '14D100P_100P',
                            '1D20P_100P', '1D1N_100P',
                            '2D100P_100P', 'UNKNOWN', '14D100P']

    interesting_nationalities = ['South Korea', 'Taiwan', 'Malaysia',
                                 'Hong Kong', 'Japan', 'China',
                                 'UNKNOWN', 'Thailand', 'Philippines',
                                 'United States of America',
                                 'Singapore', 'Indonesia', 'Australia',
                                 'Vietnam', 'United Kingdom',
                                 'India', 'Saudi Arabia', 'Russia', 'Macau',
                                 'France']

    dict1_ = {interesting_policies[i]: 2 ** i for i in
              range(len(interesting_policies) - 1, -1, -1)}
    dict2_ = {interesting_nationalities[i]: 2 ** i for i in
              range(len(interesting_nationalities) - 1, -1, -1)}

    def parse_codes(val):
        if val in dict1_.keys():
            return dict1_[val]
        return 0

    def parse_countries(val):
        if val in dict2_.keys():
            return dict2_[val]
        return 0

    def charge_preprocess(val):
        if val == "Pay Later":
            return 100
        elif val == "Pay Now":
            return 20
        else:
            return -1

    full_data["relevant_cancel"] = full_data['cancellation_policy_code'].apply(
        parse_codes)
    full_data["relevant_nationality"] = full_data[
        'customer_nationality'].apply(parse_countries)
    full_data['charge_parsed'] = full_data['charge_option'].apply(
        charge_preprocess)
    # full_data['cancellation_datetime'] = pd.to_datetime(full_data['cancellation_datetime'])
    labels = full_data["cancellation_datetime"]
    labels = labels.apply(date_to_bool)
    features = full_data.drop(
        ["cancellation_datetime", "h_booking_id", "h_customer_id",
         'cancellation_policy_code', 'customer_nationality',
         'hotel_id'],
        axis=1)
    features = features.applymap(str_to_ascii)
    return features, labels


def date_to_bool(value):
    if isinstance(value, str):
        return 1
    if math.isnan(value):
        return 0
    else:
        return 1


# math.isnan(float('nan'))

def str_to_ascii(cur_str):
    if not isinstance(cur_str, str):
        if math.isnan(cur_str):
            return 0
        return cur_str
    cur_sum = 0
    for c in cur_str:
        cur_sum += ord(c)
    return cur_sum


def evaluate_and_export(estimator: BaseEstimator, X: np.ndarray,
                        filename: str):
    """
    Export to specified file the prediction results of given estimator on given testset.

    File saved is in csv format with a single column named 'predicted_values' and n_samples rows containing
    predicted values.

    Parameters
    ----------
    estimator: BaseEstimator or any object implementing predict() method as in BaseEstimator (for example sklearn)
        Fitted estimator to use for prediction

    X: ndarray of shape (n_samples, n_features)
        Test design matrix to predict its responses

    filename:
        path to store file at

    """
    pd.DataFrame(estimator.predict(X), columns=["predicted_values"]).to_csv(filename, index=False)


def parse_test(df: pd.DataFrame):
    # only one col.
    for row in df.itertuples():
        # -> id | verdict.
        val = row[1]  # make sure working.
        # print(val)
        val = val.split("|")
        # df['id'] = val[0].strip()
        df['label'] = val[1].strip()
    return df['label']


def load_set_label(filename: str):
    """
    Load Agoda booking cancellation dataset
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector in either of the following formats:
    1) Single dataframe with last column representing the response
    2) Tuple of pandas.DataFrame and Series
    3) Tuple of ndarray of shape (n_samples, n_features) and ndarray of shape (n_samples,)
    """
    # TODO - replace below code with any desired preprocessing
    full_data = pd.read_csv(filename).drop_duplicates()
    # features = full_data[["h_booking_id",
    #                       "hotel_id",
    #                       "accommadation_type_name",
    #                       "hotel_star_rating",
    #                       "customer_nationality"]]
    full_data['booking_datetime'] = pd.to_datetime(
        full_data['booking_datetime'])
    full_data['booking_datetime_year'] = full_data['booking_datetime'].dt.year
    full_data['booking_datetime_month'] = full_data[
        'booking_datetime'].dt.month
    full_data['booking_datetime_week'] = full_data['booking_datetime'].dt.week
    full_data['booking_datetime_day'] = full_data['booking_datetime'].dt.day
    full_data['booking_datetime_hour'] = full_data['booking_datetime'].dt.hour
    full_data['booking_datetime_minute'] = full_data[
        'booking_datetime'].dt.minute
    full_data['booking_datetime_day_of_week'] = full_data[
        'booking_datetime'].dt.dayofweek
    full_data = full_data.drop(["booking_datetime"], axis=1)

    full_data['checkin_date'] = pd.to_datetime(full_data['checkin_date'])
    full_data['checkin_date_year'] = full_data['checkin_date'].dt.year
    full_data['checkin_date_month'] = full_data['checkin_date'].dt.month
    full_data['checkin_date_week'] = full_data['checkin_date'].dt.week
    full_data['checkin_date_day'] = full_data['checkin_date'].dt.day
    full_data['checkin_date_hour'] = full_data['checkin_date'].dt.hour
    full_data['checkin_date_minute'] = full_data['checkin_date'].dt.minute
    full_data['checkin_date_day_of_week'] = full_data[
        'checkin_date'].dt.dayofweek
    full_data = full_data.drop(["checkin_date"], axis=1)

    full_data['checkout_date'] = pd.to_datetime(full_data['checkout_date'])
    full_data['checkout_date_year'] = full_data['checkout_date'].dt.year
    full_data['checkout_date_month'] = full_data['checkout_date'].dt.month
    full_data['checkout_date_week'] = full_data['checkout_date'].dt.week
    full_data['checkout_date_day'] = full_data['checkout_date'].dt.day
    full_data['checkout_date_hour'] = full_data['checkout_date'].dt.hour
    full_data['checkout_date_minute'] = full_data['checkout_date'].dt.minute
    full_data['checkout_date_day_of_week'] = full_data[
        'checkout_date'].dt.dayofweek
    full_data = full_data.drop(["checkout_date"], axis=1)

    full_data['hotel_live_date'] = pd.to_datetime(full_data['hotel_live_date'])
    full_data['hotel_live_date_year'] = full_data['hotel_live_date'].dt.year
    full_data['hotel_live_date_month'] = full_data['hotel_live_date'].dt.month
    full_data['hotel_live_date_week'] = full_data['hotel_live_date'].dt.week
    full_data['hotel_live_date_day'] = full_data['hotel_live_date'].dt.day
    full_data['hotel_live_date_hour'] = full_data['hotel_live_date'].dt.hour
    full_data['hotel_live_date_minute'] = full_data[
        'hotel_live_date'].dt.minute
    full_data['hotel_live_date_day_of_week'] = full_data[
        'hotel_live_date'].dt.dayofweek
    full_data = full_data.drop(["hotel_live_date"], axis=1)

    # add dummies to parse the cancelation policy
    interesting_policies = ['1D1N_1N', '1D100P', '3D1N_1N', '3D1N_100P',
                            '3D100P', '365D100P_100P',
                            '2D100P', '3D100P_100P', '1D100P_100P',
                            '7D100P_100P', '7D1N_100P',
                            '0D0N', '7D100P', '2D1N_1N', '14D100P_100P',
                            '1D20P_100P', '1D1N_100P',
                            '2D100P_100P', 'UNKNOWN', '14D100P']

    interesting_nationalities = ['South Korea', 'Taiwan', 'Malaysia',
                                 'Hong Kong', 'Japan', 'China',
                                 'UNKNOWN', 'Thailand', 'Philippines',
                                 'United States of America',
                                 'Singapore', 'Indonesia', 'Australia',
                                 'Vietnam', 'United Kingdom',
                                 'India', 'Saudi Arabia', 'Russia', 'Macau',
                                 'France']

    dict1_ = {interesting_policies[i]: 2 ** i for i in
              range(len(interesting_policies) - 1, -1, -1)}
    dict2_ = {interesting_nationalities[i]: 2 ** i for i in
              range(len(interesting_nationalities) - 1, -1, -1)}

    def parse_codes(val):
        if val in dict1_.keys():
            return dict1_[val]
        return 0

    def parse_countries(val):
        if val in dict2_.keys():
            return dict2_[val]
        return 0

    def charge_preprocess(val):
        if val == "Pay Later":
            return 100
        elif val == "Pay Now":
            return 20
        else:
            return -1

    full_data["relevant_cancel"] = full_data['cancellation_policy_code'].apply(
        parse_codes)
    full_data["relevant_nationality"] = full_data[
        'customer_nationality'].apply(parse_countries)
    full_data['charge_parsed'] = full_data['charge_option'].apply(
        charge_preprocess)
    # full_data['cancellation_datetime'] = pd.to_datetime(full_data['cancellation_datetime'])
    features = full_data.drop(
        ["cancellation_datetime", "h_booking_id", "h_customer_id",
         'cancellation_policy_code', 'customer_nationality',
         'hotel_id'],
        axis=1)
    features = features.applymap(str_to_ascii)
    return features


if __name__ == '__main__':
    np.random.seed(0)

    # Load data
    df, cancellation_labels = load_data(
        "../datasets/agoda_cancellation_train.csv")
    estimator = AgodaCancellationEstimator()
    print("fitting")
    estimator.fit(df, cancellation_labels)
    # Fit model over data
    for i in range(1,5):
        test_X = load_set_label(f"../challenge/test_set_week_{i}.csv")
        test_Y = parse_test(pd.read_csv(f"../challenge/test_set_week_{i}_labels.csv",
                                        dtype=str))
        print(f"Loss on test_{i} was {estimator.loss(test_X,test_Y)}")


    # Store model predictions over test set
    # evaluate_and_export(estimator, test_set, "id1_id2_id3.csv")
