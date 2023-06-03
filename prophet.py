import utils as utils
from neuralprophet import NeuralProphet
import numpy as np
import pandas as pd

def prophet(df):
    X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.8)
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


prophet('four_week')