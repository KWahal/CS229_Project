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

def get_LSTM_model(df):
    train, test = utils.split_train_test(df, 0.8, split_xy=False)
    f = Forecaster(
    y=train['Auction high rate %'],
    current_dates=train['date'],
    test_length = 12,
    future_dates = 12,
    cis = False,
    )
    # f.plot()
    plt.title('Orig Series',size=16)
   # plt.show()

    figs, axs = plt.subplots(2, 1,figsize=(12,6))
    f.plot_acf(ax=axs[0],lags=36)
    f.plot_pacf(ax=axs[1],lags=36)
   # plt.show()
    stat, pval, _, _, _, _ = f.adf_test(full_res=True)
    print("stat is " + str(stat))
    print("pval is " + str(pval))

    f.set_test_length(12)       # 1. 12 observations to test the results
    f.generate_future_dates(12) # 2. 12 future points to forecast
    f.set_estimator('lstm')
    f.manual_forecast(
    lags=36,
    batch_size=32,
    epochs=15,
    validation_split=.2,
    activation='tanh',
    optimizer='Adam',
    learning_rate=0.001,
    lstm_layer_sizes=(100,)*3,
    dropout=(0,)*3,
    )
    #f.plot_test_set(ci=True)

get_LSTM_model('df_all')