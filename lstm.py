import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import utils as utils
import clean_data_final as clean
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster


TEST_LENGTH = 0.1 # original: 120
FUTURE_DATES = 200 # original: 120
EPOCHS = 30 # original: 15
LAYERS = 6 # original: 3
LEARNING_RATE = 0.001 # original: 0.001
LAGS = 5 # original: 5
BATCH_SIZE = 32 # original: 32

def get_LSTM_model_cv(df):
    df = utils.get_auction_type_df(df)
    # df = utils.resample_data(df)
    print(df)

    n_splits = 10  # Number of cross-validation folds
    tscv = TimeSeriesSplit(n_splits=n_splits)

    # mse_scores = []

    i = 0

    for train_index, test_index in tscv.split(df):
    
        data, empty = utils.split_train_test(df, 1, split_xy=False)
        
        train = data.iloc[train_index]
        test = data.iloc[test_index]
        train._append(test)

        f = Forecaster(
        y=train['Auction high rate %'],
        current_dates=train['date'],
        test_length = TEST_LENGTH,
        future_dates = FUTURE_DATES,
        cis = False,
        )
        # f.plot()
        # plt.title('Orig Series',size=16)
        # plt.show()

        # figs, axs = plt.subplots(2, 1,figsize=(12,6))
        # f.plot_acf(ax=axs[0],lags=LAGS)
        # f.plot_pacf(ax=axs[1],lags=LAGS)
        # plt.show()
        stat, pval, _, _, _, _ = f.adf_test(full_res=True)
        print("stat is " + str(stat))
        print("pval is " + str(pval))

        f.set_test_length(TEST_LENGTH)       # 1. TEST_LENGTH observations to test the results
        f.generate_future_dates(FUTURE_DATES) # 2. FUTURE_DATES future points to forecast
        f.set_estimator('lstm')
        f.save_summary_stats()
        f.manual_forecast(
        lags=LAGS,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=.2,
        activation='tanh',
        optimizer='Adam',
        learning_rate=LEARNING_RATE,
        lstm_layer_sizes=(100,)*LAYERS,
        dropout=(0,)*LAYERS,
        )
        
    # print(f.summary)
    #  ss = f.export_summary_stats('lstm')
    # print(ss)
    # print(ss.values)
    # f.save_summary_stats()
    # f.all_feature_info_to_excel(out_path='./', excel_name='lstm_feature_info.xlsx')
        f.plot_test_set(include_train=False)
        plt.savefig('images/lstm_cv/lstm' + str(i) + '.png')
        # plt.close()
        # f.plot_test_set()
        # plt.show()
        i += 1

    f.export(to_excel=True)

    # mean_mse = np.mean(mse_scores)
    # print(mean_mse)

def get_LSTM_model(df):
    df = utils.get_auction_type_df(df)
    df = utils.resample_data(df)
    train, test = utils.split_train_test(df, 1, split_xy=False)
    f = Forecaster(
    y=train['Auction high rate %'],
    current_dates=train['date'],
    test_length = TEST_LENGTH,
    future_dates = FUTURE_DATES,
    cis = False,
    )
    # f.plot()
    # plt.title('Orig Series',size=16)
    # plt.show()

    # figs, axs = plt.subplots(2, 1,figsize=(12,6))
    # f.plot_acf(ax=axs[0],lags=LAGS)
    # f.plot_pacf(ax=axs[1],lags=LAGS)
    # plt.show()
    stat, pval, _, _, _, _ = f.adf_test(full_res=True)
    print("stat is " + str(stat))
    print("pval is " + str(pval))

    f.set_test_length(TEST_LENGTH)       # 1. TEST_LENGTH observations to test the results
    f.generate_future_dates(FUTURE_DATES) # 2. FUTURE_DATES future points to forecast
    f.set_estimator('lstm')
    f.save_summary_stats()
    f.manual_forecast(
    lags=LAGS,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=LEARNING_RATE,
    lstm_layer_sizes=(100,)*LAYERS,
    dropout=(0,)*LAYERS,
    )
    
   # print(f.summary)
  #  ss = f.export_summary_stats('lstm')
   # print(ss)
   # print(ss.values)
   # f.save_summary_stats()
   # f.all_feature_info_to_excel(out_path='./', excel_name='lstm_feature_info.xlsx')
    f.plot_test_set(include_train=False)
    plt.savefig('images/lstm.png')
    plt.close()
    f.plot_test_set(include_train=False)
    plt.show()
    f.export(to_excel=True)
    

get_LSTM_model('four_week')