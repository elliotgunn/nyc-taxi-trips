import numpy as np
import pandas as pd
import category_encoders as ce

import preprocessing_functions as pf
import config

import warnings
warnings.simplefilter(action='ignore')


# Training
# ================================================

# load data
df = pf.load_data(config.PATH_TO_DATASET)

# preprocessing
df = pf.add_id(df)

df = pf.remove_invalid_neg(df, config.NEG_COLS)

df = pf.remove_invalid_zeroes(df, config.ZERO_COLS)

df = pf.remove_error_RatecodeID(df)

df = pf.convert_datetime(df, config.TIME_COLS)

train = pf.trip_length(train, 'tpep_pickup_datetime', 'tpep_dropoff_datetime')

train = pf.remove_zero_or_neg_time(train, 'trip_seconds')

# divide data into train, val, test
train = df[df.tpep_dropoff_datetime <= pd.to_datetime('2017-06-30')]
test = df[df.tpep_dropoff_datetime >= pd.to_datetime('2017-11-01')]

val = train[train.tpep_dropoff_datetime >= pd.to_datetime('2017-06-01')]
train = train[train.tpep_dropoff_datetime < pd.to_datetime('2017-06-01')]

# continue preprocessing

train = pf.extract_time(train)

train = pf.log_transform(train, config.LOG_VARS)

train = pf.to_str(train, config.VAR_TO_STR)

train = pf.reduce_cardinality(train, train)

train = pf.cat_to_str(train)

encoder = ce.OneHotEncoder(use_cat_names=True)
X_train = encoder.fit_transform(train)


# y_train
y_train = train[config.TARGET]

# train scaler and save
scaler = pf.train_scaler(X_train[config.FEATURES],
                         config.OUTPUT_SCALER_PATH)

# scale train set
X_train = scaler.transform(X_train[config.FEATURES])

# train model and save
pf.train_model(X_train,
               np.log(y_train),
               config.OUTPUT_MODEL_PATH)

print('Finished training')
