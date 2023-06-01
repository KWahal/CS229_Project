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

    model = ARIMA(train, order=(1, 1, 1)).fit()

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
    # print(test)
   # print(train)
    # model.predict(test)

    print(model(summary))

def get_arima_model_cv(df):
    data = utils.get_auction_type_df(df)
    tscv = TimeSeriesSplit(n_splits=5)

    mse_scores = []

    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Fit Auto ARIMA model
        exog_train = train_data.drop(['Auction high rate %', 'Maturity date'], axis=1)
        model = auto_arima(train_data['Auction high rate %'].astype('float64'), exogenous=exog_train.apply(pd.to_numeric), seasonal=False)
        model.fit(train_data['Auction high rate %'].astype('float64'), exogenous=exog_train.apply(pd.to_numeric))

        # Make predictions on test data
        exog_test = test_data.drop(['Auction high rate %', 'Maturity date'], axis=1).apply(pd.to_numeric)
        predictions = model.predict(n_periods=len(test_data), exogenous=exog_test)

        predictions = predictions.reset_index().drop(['index'], axis=1).iloc[:, 0]
        y_test = test_data['Auction high rate %'].astype('float64').squeeze().reset_index().drop(['index'], axis=1)['Auction high rate %']

        # Calculate mean squared error
        mse = mean_squared_error(y_test, predictions)
        mse_scores.append(mse)

        # Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the first series
        #ax.plot(train['Auction high rate %'], label='predictions')
        ax.plot(predictions, label='predictions')

        # Plot the second series
        #ax.plot(model.predict_in_sample(exogenous=exog_train), label='test')
        ax.plot(y_test, label='test')

        # Set labels and title
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_title('Line Plot of Two Series')

        # Display a legend
        ax.legend()

        # Show the plot
        plt.show()  

    mean_mse = np.mean(mse_scores)
    print(mean_mse)
        


def get_arima_model_char(df):
    # Prepare train and test data
    train, test = utils.split_train_test(df, 0.5, split_xy=False)
   # print(train.reset_index().drop(['index'], axis=1))
    #print(test.reset_index().drop(['index'], axis=1))
    # Fit ARIMA model on training data
    exog_train = train.drop(['Auction high rate %', 'Maturity date'], axis=1)
   
    model = auto_arima(train['Auction high rate %'].astype('float64'), exogenous=exog_train.apply(pd.to_numeric), seasonal=False)
    model.fit(train['Auction high rate %'].astype('float64'), exogenous=exog_train.apply(pd.to_numeric))

    # Predict on test data
    exog_test = test.drop(['Auction high rate %', 'Maturity date'], axis=1).apply(pd.to_numeric)
    predictions = model.predict(n_periods=len(test), exogenous=exog_test)
    
    predictions = predictions.reset_index().drop(['index'], axis=1).iloc[:, 0]
    y_test = test['Auction high rate %'].astype('float64').squeeze().reset_index().drop(['index'], axis=1)['Auction high rate %']

    # Evaluate model performance
    mse = ((predictions - y_test) ** 2).mean()
    rmse = mse ** 0.5

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Plot the first series
    #ax.plot(train['Auction high rate %'], label='predictions')
    ax.plot(predictions, label='predictions')

    # Plot the second series
    #ax.plot(model.predict_in_sample(exogenous=exog_train), label='test')
    ax.plot(y_test, label='test')

    # Set labels and title
    ax.set_xlabel('X-axis')
    ax.set_ylabel('Y-axis')
    ax.set_title('Line Plot of Two Series')

    # Display a legend
    ax.legend()

    # Show the plot
    plt.show()  

def evaluate_models(p, d, q):
    rmse = 1000000

    for i in range(p):
        for j in range(d):
            for k in range(q):
                temp = get_arima_model_char('four week', p, d, q)
                if temp < rmse:
                    rmse = temp
                    best_p = i
                    best_d = j
                    best_q = k
    
    print(rmse)
    print(best_p)
    print(best_d)
    print(best_q)

get_arima_model_cv('cmb')