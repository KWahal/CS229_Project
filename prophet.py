import utils as utils
from neuralprophet import NeuralProphet, set_log_level
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prophet(df):
    df = utils.get_auction_type_df(df)
    df_resampled = utils.resample_data(df)
    X_train, y_train, X_test, y_test = get_split(df_resampled, 0.8, split_xy=True)

    X_train = X_train[:, 0]
    X_test = X_test[:, 0]

    data = np.column_stack((X_train, y_train))
    data = pd.DataFrame(data)
    data = data.rename(columns={0: "ds", 1: "y"})

    test_data = np.column_stack((X_test, y_test))
    test_data = pd.DataFrame(test_data)
    test_data = test_data.rename(columns={0: "ds", 1: "y"})

    # plot existing data
    """
    x = data.iloc[:, 0]  # Selecting the first column as x-axis values
    y = data.iloc[:, 1]  # Selecting the second column as y-axis values

    plt.plot(x, y)
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('Line Plot')
    plt.show()"""

    # Create a NeuralProphet model with default parameters
    m = NeuralProphet(yearly_seasonality=False,
    weekly_seasonality=False,
    daily_seasonality=False,)
    set_log_level("ERROR")

    # Fit the model on the dataset (this might take a bit)
    metrics = m.fit(data)

    prediction = m.predict(test_data)
    m.plot(prediction).show()
    m.plot_parameters(components=["trend"]).show()

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