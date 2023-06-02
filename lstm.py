import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import utils as utils
import clean_data_final as clean
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scalecast.Forecaster import Forecaster


TEST_LENGTH = 120 # original: 120
FUTURE_DATES = 1200 # original: 120
EPOCHS = 15 # original: 15
LAYERS = 6 # original: 3

def get_LSTM_model(df):
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
    # f.plot_acf(ax=axs[0],lags=36)
    # f.plot_pacf(ax=axs[1],lags=36)
    # plt.show()
    stat, pval, _, _, _, _ = f.adf_test(full_res=True)
    print("stat is " + str(stat))
    print("pval is " + str(pval))

    f.set_test_length(TEST_LENGTH)       # 1. TEST_LENGTH observations to test the results
    f.generate_future_dates(FUTURE_DATES) # 2. FUTURE_DATES future points to forecast
    f.set_estimator('lstm')
    f.save_summary_stats()
    f.manual_forecast(
    lags=36,
    batch_size=32,
    epochs=EPOCHS,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=0.001,
    lstm_layer_sizes=(100,)*LAYERS,
    dropout=(0,)*LAYERS,
    )
    ss = f.export_summary_stats('lstm')
    print(ss)
    print(ss.values)
    f.plot_test_set()
    plt.savefig('images/lstm.png')
    f.plot_test_set()
    plt.show()
    

get_LSTM_model('df_all')