import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import charlie_utils as utils

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
    x_train, y_train, x_test, y_test = utils.split_train_test(df, 0.7)
    x_train = np.delete(x_train, 0, axis=1) # delete the time data
    x_test = np.delete(x_test, 0, axis=1)

    # Fit the regression
    model = LinearRegression()
    model.fit(x_train, y_train)
    model.predict(x_test)
    r_sq = model.score(x_test, y_test)
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

def get_arima_model(df):
    train, test = utils.split_train_test(df, 0.7, split_xy=False)

    train['date'] = pd.to_datetime(train['date'])
    train = train.set_index('date')

    model = sm.tsa.ARIMA(train, order=(1, 1, 1)).fit()

    model.predict(test)

    print(model(summary))

get_arima_model('four_week')