import pandas as pd

import joblib
import config


def make_prediction(input_data):

    # load pipeline with all the transformation steps
    _pipe_tip = joblib.load(filename=config.PIPELINE_NAME)

    # get predictions
    results = _pipe_tip.predict(input_data)

    return results

if __name__ == '__main__':

    # test pipeline
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score

    data = pd.read_csv(config.TRAINING_DATA_FILE)

    # divide train and test
    X_train, X_test, y_train, y_test = train_test_split(
        data[config.FEATURES],
        data[config.TARGET],
        test_size=0.1,
        random_state=1)

    pred = make_prediction(X_test)

    # determine mse and rmse
    print(f'test mse: {int(
        mean_squared_error(y_test, np.exp(pred)))}')
    print(f'test rmse: {int(
        np.sqrt(mean_squared_error(y_test, np.exp(pred))))}')
    print(f'test r2: {r2_score(y_test, np.exp(pred))}')
    print()
