import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import soham_utils as utils
import clean_data as clean
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

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
    x_train, y_train, x_test, y_test = utils.split_train_test(df, 0.8)
    x_train = np.delete(x_train, 0, axis=1) # delete the time data
    x_test = np.delete(x_test, 0, axis=1)

    # Fit the regression
    model = LinearRegression()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    r_sq = model.score(x_test, y_test)
    
    MSE = mean_squared_error(y_test, y_pred)

    print(f"MSE: {MSE}")
    print(f"coefficient of determination: {r_sq}")
    print(f"intercept: {model.intercept_}")
    print(f"slope: {model.coef_}")

def get_polynomial_regression_model(df):
    x_train, y_train, x_test, y_test = utils.split_train_test(df, 0.8)
    x_train = np.delete(x_train, 0, axis=1) # delete the time data
    x_test = np.delete(x_test, 0, axis=1)

    degree=3
    polyreg_scaled = make_pipeline(PolynomialFeatures(degree), StandardScaler(), LinearRegression())
    polyreg_scaled.fit(x_train, y_train)

    y_pred = polyreg_scaled.predict(x_test)
    r_sq = polyreg_scaled.score(x_test, y_test)
    
    MSE = mean_squared_error(y_test, y_pred)

    print(f"MSE: {MSE}")
    print(f"coefficient of determination: {r_sq}")
   # print(f"intercept: {polyreg_scaled.intercept_}")
   # print(f"slope: {polyreg_scaled.coef_}")

    #polyreg=make_pipeline(PolynomialFeatures(degree),LinearRegression())
    #polyreg.fit(x_train,y_train)
    

def get_arima_model(df):
    train, test = utils.split_train_test(df, 0.8, split_xy=False)

    train['date'] = pd.to_datetime(train['date'])
    train = train.set_index('date')

    train = train.drop('Maturity date', axis=1)
   # print(train)
   # print(train.dtypes)

    for col in train.columns:
        train[col] = train[col].astype('float64')
       # if col != 'Auction high rate %':
            #train = train.drop('Auction high rate %', axis=1)

    train = train.drop('Total issue', axis = 1)
    train = train.drop('(SOMA) Federal Reserve banks', axis=1)
    train = train.drop('Depository institutions', axis=1)
    train = train.drop('Individuals', axis = 1)
    train = train.drop('Dealers and brokers', axis=1)
    train = train.drop('Pension and Retirement funds and Ins. Co.', axis=1)
    train = train.drop('Investment funds', axis = 1)
    train = train.drop('Foreign and international', axis=1)
    train = train.drop('Other and Noncomps', axis=1)
    train = train.drop('News Sentiment', axis=1)
   # print(train)

    model = sm.tsa.ARIMA(train, order=(1, 1, 1)).fit()

    print(model.summary())

    test['date'] = pd.to_datetime(test['date'])
    #test = test.set_index('date')

    print(test)
    test = test.drop('Maturity date', axis=1)
    test = test.drop('Total issue', axis = 1)
    test = test.drop('(SOMA) Federal Reserve banks', axis=1)
    test = test.drop('Depository institutions', axis=1)
    test = test.drop('Individuals', axis = 1)
    test = test.drop('Dealers and brokers', axis=1)
    test = test.drop('Pension and Retirement funds and Ins. Co.', axis=1)
    test = test.drop('Investment funds', axis = 1)
    test = test.drop('Foreign and international', axis=1)
    test = test.drop('Other and Noncomps', axis=1)
    test = test.drop('News Sentiment', axis=1)

    test['Auction high rate %'] = test['Auction high rate %'].astype('float64')
    
    for col in test.columns:
        print("column is")
        print(col)
    print(test.dtypes)
    print(test)
    test = test['date']
   # print(test['date'])
    #print(test)
   # print(train)
    model.predict(test)

    print(model(summary))

# get_polynomial_regression_model('df_all')
#get_linear_regression_model('df_all')
get_arima_model('four_week')