PATH_TO_DATASET = 'sample.csv'
OUTPUT_SCALER_PATH = 'scaler.pkl'
OUTPUT_MODEL_PATH = 'lasso_regression.pkl'

# preprocessing features
TARGET = 'tip_amount'

NEG_COLS = ['fare_amount', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge', 'total_amount']

ZERO_COLS = ['fare_amount', 'total_amount']

TIME_COLS = ['tpep_pickup_datetime', 'tpep_dropoff_datetime']

LOG_VARS = ['fare_amount', 'total_amount']

VAR_TO_STR = ['VendorID', 'RatecodeID', 'PULocationID', 'DOLocationID', 'store_and_fwd_flag', 'payment_type']
