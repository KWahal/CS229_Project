import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA
import utils as utils
import clean_data_final as clean_data
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib.pyplot as plt
from pmdarima.arima import auto_arima
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

# To run a model on a specified auction type, pick from the following as the parameter for df:
# 'df_all'
# 'four_week'
# 'eight_week'
# 'thirteen_week'
# 'seventeen_week'
# 'twenty_six_week'
# 'fifty_two_week'
# 'cmb'

def get_linear_regression_model(df):
    data = utils.get_auction_type_df(df)
    data = utils.resample_data(data)
    x_train, y_train, x_test, y_test = utils.split_train_test(data, 0.8)
    x_train = np.delete(x_train, 0, axis=1) # delete the time data
    x_test = np.delete(x_test, 0, axis=1)

    # Fit the regression
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r_sq = model.score(x_test, y_test)

    y_pred_train = model.predict(x_train)
    MSE_train = mean_squared_error(y_train, y_pred_train)
    
    MSE = mean_squared_error(y_test, y_pred)

    print(f"MSE train: {MSE_train}")
    print(f"MSE: {MSE}")
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Linear Regression Model: True Values vs. Predicted Values')
    plt.legend()
    plt.show()

get_linear_regression_model('four_week')