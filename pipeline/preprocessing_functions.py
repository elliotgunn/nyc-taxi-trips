import pandas as pd
import numpy as np

# to divide train and test set
from sklearn.model_selection import train_test_split

# feature scaling
from sklearn.preprocessing import StandardScaler

# to build the models
from sklearn.linear_model import LinearRegression, Lasso

# to evaluate the models
from sklearn.metrics import mean_squared_error


def load_data(df_path):
    return pd.read_csv(df_path)

### need function for removing negative/zero values from columns

def train_test(df, target):
    X_train, X_test, y_train, y_test = train_test_split(df, df.target,
                                                    test_size=0.1,
                                                    random_state=1)
    return X_train, X_test, y_train, y_test

def trip_time(df, start, end):
    """
    start: pickup time
    end: dropoff time

    Extracts time information from the two variables.
    """
    df = df.copy()

    df['pickup'] = pd.to_datetime(df[start])
    df['dropoff'] = pd.to_datetime(df[end])

    df['trip_time'] = df['dropoff'] - df['pickup']

    # convert to minutes
    df['trip_seconds'] = df['trip_time'].astype('timedelta64[s]')

    df = df.drop(columns=[start, end, 'trip_time', 'pickup', 'dropoff'])

    return df

def log_transform(df, var):
    return np.log(df[var])

def to_str(train, test, var):
    train[var] = train[var].astype(str)
    test[var] = test[var].astype(str)

## add def reduce_cardinality

## add def categorical_encoders
