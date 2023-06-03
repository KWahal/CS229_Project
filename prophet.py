import utils as utils
from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd

def prophet(df):
    #resampled_data = utils.resample_data(df)

    X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.8)

    X_train = utils.resample_data(X_train)
    print(X_train)

    X_train = X_train[:, 0]
    print(y_train.shape)
    print(X_train.shape)
    data = np.column_stack((X_train, y_train))
    data = pd.DataFrame(data)
    data = data.rename(columns={0: "ds", 1: "y"})
    print(data)

    m = NeuralProphet(n_forecasts = 60, n_lags=60, n_changepoints=50, yearly_seasonality=False, weekly_seasonality=False, 
                      daily_seasonality=False, batch_size=64, epochs=50, learning_rate=1.0)
    
    metrics = m.fit(data, freq="D")

def get_split(df, split_size, split_xy=True):
    ts = df

    # Split into a training set and a testing set
    train_size = int(len(ts) * split_size)
    train_ts, test_ts = ts[:train_size], ts[train_size:]

    # Splits training and testing sets into x and y
    x_train, y_train = utils.create_arrays(train_ts)
    x_test, y_test = utils.create_arrays(test_ts)

    if (split_xy):
        return x_train, y_train, x_test, y_test
    return train_ts, test_ts


prophet('four_week')