import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import Lasso

# import joblib


# preprocessing
# ================================================

def load_data(df_path):
    return pd.read_csv(df_path)


def add_id(df):
    df = df.copy()
    df['id'] = df.index + 1

    return df


def remove_invalid_neg(df, cols):
    """
    Remove negative values columns where it is not possible to have negative values
    """
    for col in cols:
        df = df[df[col] >= 0]

    return df


def remove_invalid_zeroes(df, cols):
    """
    Remove zero values from where it is not possible to have zero values
    """
    for col in cols:
        df = df[df[col] > 0]

    return df


def remove_error_RatecodeID(df):
    """
    Remove the one erroenous coding instance
    """

    df = df[df.RatecodeID != 99]

    return df


def convert_datetime(df, cols):
    """
    Extracts time information from the datetime variables.
    """
    df = df.copy()

    for col in cols:
        df[col] = pd.to_datetime(df[col])

    return df


def trip_length(df, start, end):
    """
    start: pickup time
    end: dropoff time

    Extracts time information from the two variables.
    """
    df = df.copy()

    df['trip_time'] =  df[end] - df[start]

    # convert to seconds
    df['trip_seconds'] = df['trip_time'].astype('timedelta64[s]')

    df = df.drop(columns=['trip_time'])

    return df


def remove_zero_or_neg_time(df, col):
    """
    In the feature engineered `trip_seconds` column,
    there are negative or zero values to be removed.
    """

    df = df[df[col] > 0]

    return df


def extract_time(df):
    """
    Extract:
    - hour, min, second of the day
    - day of the month
    """

    df['pickup_hour'] = df['tpep_pickup_datetime'].dt.hour
    df['pickup_min'] = df['tpep_pickup_datetime'].dt.minute
    df['pickup_sec'] = df['tpep_pickup_datetime'].dt.second

    df['dropoff_hour'] = df['tpep_dropoff_datetime'].dt.hour
    df['dropoff_min'] = df['tpep_dropoff_datetime'].dt.minute
    df['dropoff_sec'] = df['tpep_dropoff_datetime'].dt.second

    df['trip_day'] = df['tpep_dropoff_datetime'].dt.day

    df = df.drop(columns=['tpep_pickup_datetime', 'tpep_dropoff_datetime'])

    return df


def log_transform(df, cols):
    """
    log transform numeric variables that do not contain zeros
    """

    for col in cols:
        df[col] = np.log(df[col])

    return df


def to_str(df, cols):

    for col in cols:
        df[col] = df[col].astype(str)

    return df


def reduce_cardinality(df):
    """
    Need to reduce cardinality of PULocationID & DOLocationID
    """

    # get list of top 10 locations
    top10PU = df['PULocationID'].value_counts()[:9].index
    top10DO = df['DOLocationID'].value_counts()[:9].index

    # for locationIDs not in the top 10, replace with OTHER
    df.loc[~df['PULocationID'].isin(top10PU), 'PULocationID'] = 'OTHER'
    df.loc[~df['DOLocationID'].isin(top10DO), 'DOLocationID'] = 'OTHER'

    return df


def cat_to_str(df):

    df['RatecodeID'] = df['RatecodeID'].replace({'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six'})
    df['RatecodeID'] = df['RatecodeID'].replace({'1': 'one', '2': 'two', '3': 'three', '4': 'four', '5': 'five', '6': 'six'})

    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'1': 'one', '2': 'two'})
    df['store_and_fwd_flag'] = df['store_and_fwd_flag'].replace({'1': 'one', '2': 'two'})

    df['payment_type'] = df['payment_type'].replace({'1': 'one', '2': 'two', '3': 'three', '4': 'four'})
    df['payment_type'] = df['payment_type'].replace({'1': 'one', '2': 'two', '3': 'three', '4': 'four'})

    return df


# training functions
# ================================================

def train_scaler(df, output_path):
    scaler = MinMaxScaler()
    scaler.fit(df)
    joblib.dump(scaler, output_path)
    return scaler



def scale_features(df, scaler):
    scaler = joblib.load(scaler) # with joblib probably
    return scaler.transform(df)



def train_model(df, target, output_path):
    # initialise the model
    lin_model = Lasso(alpha=0.005, random_state=0)

    # train the model
    lin_model.fit(df, target)

    # save the model
    joblib.dump(lin_model, output_path)

    return None


def predict(df, model):
    model = joblib.load(model)
    return model.predict(df)
