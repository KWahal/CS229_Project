"""from numpy import array
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D


def prep_data():
    data = {}
    for filename in os.listdir(DATADIR):
        if not filename.lower().endswith(".csv"):
            continue # read only the CSV files
        filepath = os.path.join(DATADIR, filename)
        X = pd.read_csv(filepath, index_col="Date", parse_dates=True)
        # basic preprocessing: get the name, the classification
        # Save the target variable as a column in dataframe for easier dropna()
        name = X["Name"][0]
        del X["Name"]
        cols = X.columns
        X["Target"] = (X["Close"].pct_change().shift(-1) > 0).astype(int)
        X.dropna(inplace=True)
        # Fit the standard scaler using the training dataset
        index = X.index[X.index > TRAIN_TEST_CUTOFF]
        index = index[:int(len(index) * TRAIN_VALID_RATIO)]
        scaler = StandardScaler().fit(X.loc[index, cols])
        # Save scale transformed dataframe
        X[cols] = scaler.transform(X[cols])
        data[name] = X
def cnnpred_2d(seq_len=60, n_features=82, n_filters=(8,8,8), droprate=0.1):
    "2D-CNNpred model according to the paper"
    model = Sequential([
        Input(shape=(seq_len, n_features, 1)),
        Conv2D(n_filters[0], kernel_size=(1, n_features), activation="relu"),
        Conv2D(n_filters[1], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Conv2D(n_filters[2], kernel_size=(3,1), activation="relu"),
        MaxPool2D(pool_size=(2,1)),
        Flatten(),
        Dropout(droprate),
        Dense(1, activation="sigmoid")
    ])
    return model"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns 
import matplotlib.pyplot as plt
from numpy import array

import tensorflow as tf
from tensorflow.keras.models import Sequential

from keras.layers import Dense,RepeatVector
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection import TimeSeriesSplit
import utils as utils
import clean_data_final as clean
from matplotlib import pyplot


def clean_data(X_train, y_train, X_test, y_test):
    for final_array in [X_train, X_test, y_test]:
        dates = final_array[0, :]
        difference = pd.Series(dates)-dates[0]
        df_diff = pd.DataFrame(difference) #find diff since original date
        df_diff['weeks'] = df_diff / pd.Timedelta(weeks=1) # convert to weeks
        weeks_array = pd.DataFrame(df_diff['weeks'])
        weeks_array = np.array(weeks_array.values.tolist()) # convert back to numpy


        final_array = np.delete(final_array, 0, axis=1) # remove the original datetime column
    return X_train, y_train, X_test, y_test
"""
def cnn(df):
   # data = pd.read_csv('data.csv')

    ### BARD
    # Split the data into train and test sets
    X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.8)

    X_train, y_train, X_test, y_test = clean_data(X_train, y_train, X_test, y_test)
    # Assuming you have your input data X and target variable y

    # Pad the input data to increase the input size
    # Define the input shape based on the length of the time series
    input_shape = (X_train.shape[1], 1)

    # Reshape the input data to match the expected input shape of the CNN
    X_train = X_train.reshape(X_train.shape[0], input_shape[0], input_shape[1])
    X_test = X_test.reshape(X_test.shape[0], input_shape[0], input_shape[1])

    # Create the CNN model
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer with 1 neuron for regression

    # Compile the model
    model.compile(loss='mse', optimizer='adam')

    # Train the model
    model.fit(X_train, y_train, epochs=10, batch_size=32)

    # Evaluate the model on the test data
    loss = model.evaluate(X_test, y_test)

    # Once the model is trained, you can use it for prediction
    predictions = model.predict(X_test)

    # Once the model is trained, you can use it for prediction
    # Assuming you have a new data sample X_test for prediction
    X_test_padded = np.pad(X_test, ((0, 0), (padding_size, padding_size)), mode='constant')
    X_test_padded = np.reshape(X_test_padded, (X_test_padded.shape[0], X_test_padded.shape[1], 1))
    predictions = model.predict(X_test_padded)


    # Reshape the data for Keras
   # X_train = X_train.values.reshape(-1, 1, 12)
    #X_test = X_test.values.reshape(-1, 1, 12)

    # Create the model
    model = Sequential()
    model.add(Conv1D(32, 3, activation='relu', input_shape=(1, 12)))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=10)

    # Evaluate the model
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

    ### END BARD
    data = utils.split_train_test(df, 1, split_xy=False)[0]
    
    timestep = 30

    X= []
    Y=[]

    raw_data=data

    for i in range(len(raw_data)- (timestep)):
        X.append(raw_data[i:i+timestep])
        Y.append(raw_data[i+timestep])


    X=np.asanyarray(X)
    Y=np.asanyarray(Y)


    k = 850
    print()
    Xtrain = X[:k,:,:]  
    Ytrain = Y[:k]    

    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(30, 1)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(Xtrain, Ytrain, epochs=200, verbose=0)"""

def cnn(df):
    df=utils.get_auction_type_df(df)
    X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.92)
    print(X_train.shape)
    print(X_train)
   # print(X_train.dtypes)
   # X_train, y_train, X_test, y_test = clean_data(X_train, y_train, X_test, y_test)
    """timestamps = X_train[:, 0]
    timestamps2 = X_test[:, 0]

# Calculate the time differences by subtracting the first element from the entire column
    time_diff = timestamps - timestamps[0]
    time_diff2 = timestamps2 - timestamps2[0]

# Convert the time differences to numeric representation (e.g., seconds)
    time_diff_numeric = time_diff.astype('timedelta64[s]')
    time_diff_numeric_2 = toime_diff2.astype('timedelta64[s]')

# Replace the first column in 'X' with the numeric representation
    X_train[:, 0] = time_diff_numeric
    X_test[:, 0] = time_diff_numeric_2"""
    # Convert the first column of 'X' to a Pandas series
    
    timestamps = pd.Series(X_train[:, 0])
    timestamps2 = pd.Series(X_test[:, 0])

    # Calculate the time differences by subtracting the first element from the entire series
    time_diff = (timestamps - timestamps[0])/pd.Timedelta(weeks=1)
    time_diff2 = (timestamps2 - timestamps2[0])/pd.Timedelta(weeks=1)

    # Replace the first column in 'X' with the numeric representation
    X_train[:, 0] = time_diff
    X_test[:, 0] = time_diff2

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    y_train = y_train.astype('float32')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    print(X_train.shape)
    print(y_train.shape)


    input_shape = (X_train.shape[0], X_train.shape[1])  # (num_time_steps, num_features)

    """
    # Create the model
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape, padding='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Adjust the output layer according to your task
  #  model.add(tf.keras.layers.Dense(256, input_shape=(X_train.shape[1],), activation='sigmoid'))

    # Compile the model
    model.compile(optimizer='adam', loss='mse')"""
    model = Sequential()
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(11, 1)))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(64, activation='relu'))  # Additional dense layer
    model.add(Dense(32, activation='relu'))
   # model.add(Dense(16, activation='relu')) # better without this
    #model.add(Dense(8, activation='relu')) # better without i think
    model.add(Dense(1)) # add more dense layers, do more on the learning_rate
    model.compile(optimizer='adam', loss='mse')
    # fit model
    model.fit(X_train, y_train, epochs=200)

    # Train the model
   # print(X_train)
    print(model.summary())
    preds = model.predict(X_test)
   # scaler = MinMaxScaler(feature_range=(-1,1))
  #  preds = scaler.inverse_transform(preds)


    #Ytest=np.asanyarray(y_test)  
    Ytest=y_test.reshape(-1,1) 
    
   # Ytest = scaler.inverse_transform(Ytest)


   # Ytrain=np.asanyarray(Ytrain)  
    Ytrain=y_train.reshape(-1,1) 
   # Ytrain = scaler.inverse_transform(Ytrain)

    mse = mean_squared_error(y_test,preds)
    print("mse is " + str(mse))

    Ytest_df = pd.DataFrame(Ytest, columns = ['Value'])
    Ytest_df.insert(0, 'Date', timestamps2.tolist(), True)
    Ytest_df = Ytest_df.groupby('Date')['Value'].mean().reset_index()

    preds_df = pd.DataFrame(preds, columns = ['Value'])
    preds_df.insert(0, 'Date', timestamps2.tolist(), True)
    preds_df = preds_df.groupby('Date')['Value'].mean().reset_index()

    # plot
    pyplot.figure(figsize=(20,10))
    pyplot.plot(Ytest_df['Date'], Ytest_df['Value'])
    pyplot.plot(preds_df['Date'], preds_df['Value'], 'r')
    pyplot.savefig('images/cnn.png')
    pyplot.show()
   # model.fit(X_train, y_train, epochs=10)
   # model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

def cnn_crossval(df):
    df=utils.get_auction_type_df(df)
    X_train, y_train, X_test, y_test = utils.split_train_test(df, 0.92)

    timestamps = pd.Series(X_train[:, 0])
    timestamps2 = pd.Series(X_test[:, 0])

    # Calculate the time differences by subtracting the first element from the entire series
    time_diff = (timestamps - timestamps[0])/pd.Timedelta(weeks=1)
    time_diff2 = (timestamps2 - timestamps2[0])/pd.Timedelta(weeks=1)

    # Replace the first column in 'X' with the numeric representation
    X_train[:, 0] = time_diff
    X_test[:, 0] = time_diff2

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    y_test = y_test.astype('float32')
    y_train = y_train.astype('float32')
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Merge inputs and targets
    inputs = np.concatenate((X_train, X_test), axis=0)
    targets = np.concatenate((y_train, y_test), axis=0)

    # Define the K-fold Cross Validator
    num_folds = 10
    kfold = KFold(n_splits=num_folds, shuffle=True)

    # K-fold Cross Validation model evaluation
    fold_no = 1
    for train, test in kfold.split(inputs, targets):

        # Define the model architecture
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(11, 1)))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(64, activation='relu'))  # Additional dense layer
        model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu')) # better without this
        #model.add(Dense(8, activation='relu')) # better without i think
        model.add(Dense(1)) # add more dense layers, do more on the learning_rate

        # Compile the model
        model.compile(optimizer='adam', loss='mse')


        # Generate a print
        print('------------------------------------------------------------------------')
        print(f'Training for fold {fold_no} ...')

        # Fit data to model
        history = model.fit(inputs[train], targets[train], epochs=200)

        # Generate generalization metrics
       # scores = model.evaluate(inputs[test], targets[test], verbose=0)
       # print(f'Score for fold {fold_no}: {model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')
      #  acc_per_fold.append(scores[1] * 100)
       # loss_per_fold.append(scores[0])

        # Increase fold number
        fold_no = fold_no + 1

def cnn_crossval_arima(df):
    data = utils.get_auction_type_df(df)
  #  data = utils.resample_data(data)
    df = pd.DataFrame(data)

# Extract the first column as a pandas Series
    timestamps = df.iloc[:, 0]

    # Calculate the time differences by subtracting the first element from the entire series
    time_diff = np.array((timestamps - timestamps[0])/pd.Timedelta(weeks=1))

    # Replace the first column in 'X' with the numeric representation
    data['date'] = time_diff
    data = data.drop('Maturity date', axis=1)

   # data = np.hstack((time_diff.values.reshape(-1, 1), data[:, 1:]))
    data = data.astype('float32')

   # X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
   # X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        # Merge inputs and targets
   # inputs = np.concatenate((X_train, X_test), axis=0)
   # targets = np.concatenate((y_train, y_test), axis=0)

    tscv = TimeSeriesSplit(n_splits=10)

    mse_scores = []

    i = 0

    for train_index, test_index in tscv.split(data):
        train_data = data.iloc[train_index]
        test_data = data.iloc[test_index]

        # Fit model
        model = Sequential()
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu', input_shape=(18, 1)))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(Conv1D(filters=128, kernel_size=2, activation='relu'))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(100, activation='relu'))
        model.add(Dense(64, activation='relu'))  # Additional dense layer
        model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu')) # better without this
        #model.add(Dense(8, activation='relu')) # better without i think
        model.add(Dense(1)) # add more dense layers, do more on the learning_rate

        # Compile the model
        model.compile(optimizer='adam', loss='mse')

        targets = pd.DataFrame(train_data['Auction high rate %']).to_numpy()
        inputs = pd.DataFrame(train_data.drop('Auction high rate %', axis=1)).to_numpy()

       # inputs = train_data[]
       # inputs = pd.DataFrame(train_data.iloc[:, -1]).to_numpy()
      #  targets = pd.DataFrame(train_data.iloc[:, :-1]).to_numpy()
      #  print(inputs)
       # print(targets)
        history = model.fit(inputs, targets, epochs=200)

        # Make predictions on test data
        #test_inputs = test_data.iloc[:, -1]
        #y_test = test_data.iloc[:, :-1]
        y_test = pd.DataFrame(test_data['Auction high rate %']).to_numpy()
        test_inputs = pd.DataFrame(test_data.drop('Auction high rate %', axis=1)).to_numpy()
        preds = model.predict(test_inputs)

        # Calculate mean squared error
        Ytest=y_test.reshape(-1,1) 

        mse = mean_squared_error(y_test,preds)
        print("mse is " + str(mse))
        mse_scores.append(mse)

        Ytest_df = pd.DataFrame(Ytest, columns = ['Value'])
        Ytest_df.insert(0, 'Date', timestamps.iloc[test_index].tolist(), True)
        Ytest_df = Ytest_df.groupby('Date')['Value'].mean().reset_index()

        preds_df = pd.DataFrame(preds, columns = ['Value'])
        preds_df.insert(0, 'Date', timestamps.iloc[test_index].tolist(), True)
        preds_df = preds_df.groupby('Date')['Value'].mean().reset_index()

        # plot
        pyplot.figure(figsize=(20,10))
        pyplot.plot(Ytest_df['Date'], Ytest_df['Value'])
        pyplot.plot(preds_df['Date'], preds_df['Value'], 'r')
        pyplot.xlabel("Dates")
        pyplot.ylabel("Values")
        pyplot.savefig('images/cnn_cv/cnn' + str(i) + '.png')
        pyplot.show()
        i += 1
    mean_mse = np.mean(mse_scores)
    print("mean mse is " + str(mean_mse))

cnn('df_all')
#cnn_crossval('df_all')
# cnn_crossval_arima('df_all')

