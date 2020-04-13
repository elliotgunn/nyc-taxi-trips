import preprocessing_functions as pf
import config
import category_encoders as ce
import pandas as pd


# =========== scoring pipeline =========

def predict(df):
    print('predict function b4 anything', df.shape)


    print('predict function', df.shape)

    df = pf.extract_time(df)

    df = pf.log_transform(df, config.LOG_VARS)

    df = pf.to_str(df, config.VAR_TO_STR)

    df = pf.reduce_cardinality(df, df)

    df = pf.cat_to_str(df)

    encoder = ce.OneHotEncoder(use_cat_names=True)
    df = encoder.fit_transform(df)


    # scale variables
    df = pf.scale_features(df[config.FEATURES],
                             config.OUTPUT_SCALER_PATH)


    # make predictions
    predictions = pf.predict(df, config.OUTPUT_MODEL_PATH)

    return predictions



if __name__ == '__main__':

    from math import sqrt
    import numpy as np

    from sklearn.metrics import mean_squared_error, r2_score

    import warnings
    warnings.simplefilter(action='ignore')

    # Load data
    df = pf.load_data(config.PATH_TO_DATASET)

    df = pf.add_id(df)
    df = pf.remove_invalid_neg(df, config.NEG_COLS)
    df = pf.remove_invalid_zeroes(df, config.ZERO_COLS)
    df = pf.remove_error_RatecodeID(df)
    df = pf.convert_datetime(df, config.TIME_COLS)

    df = pf.trip_length(df, 'tpep_pickup_datetime', 'tpep_dropoff_datetime')
    df = pf.remove_zero_or_neg_time(df, 'trip_seconds')

    # train/val/test split
    train = df[df.tpep_dropoff_datetime <= pd.to_datetime('2017-06-30')]
    test = df[df.tpep_dropoff_datetime >= pd.to_datetime('2017-11-01')]

    val = train[train.tpep_dropoff_datetime >= pd.to_datetime('2017-06-01')]
    train = train[train.tpep_dropoff_datetime < pd.to_datetime('2017-06-01')]


    y_val = val[config.TARGET]
    print('df', df.shape)
    print('val', val.shape)
    print('y_val', y_val.shape)


    # predict
    pred = predict(val)

    print('pred', pred.shape)

    # determine mse and rmse
    print('val mse: {}'.format(int(
        mean_squared_error(y_val, np.exp(pred)))))
    print('val rmse: {}'.format(int(
        sqrt(mean_squared_error(y_val, np.exp(pred))))))
    print('val r2: {}'.format(
        r2_score(y_val, np.exp(pred))))
    print()
