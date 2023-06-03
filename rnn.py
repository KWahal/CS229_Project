import numpy as np
import pandas as pd
import utils as utils

def get_rnn(df):
    df = utils.get_auction_type_df(df)
    df_resampled = utils.resample_data(df)
    print(df_resampled)

    #X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.7, split_xy=True)
    #print(X_train)
    #print(y_train)
    #print(X_test)
    #print(y_test)
    #model = Sequential()
    #model.add(LSTM(units=64, input_shape=(timesteps, num_features)))
    #model.add(Dense(units=1))

    #model.compile(loss='mean_squared_error', optimizer='adam')

    #model.fit(X_train, y_train, batch_size=32, epochs=10)

    #y_pred = model.predict(X_test)
    #mse = np.mean((y_pred - y_test) ** 2)

    #future_predictions = model.predict(X_future)

get_rnn('four_week')