import pandas as pd
import numpy as np
import charlie_clean_data as clean_data
import matplotlib.pyplot as plt
import math

def get_auction_type_df(df):
    df_all = clean_data.clean_main_data()
    if (df == 'df_all'):
        return df_all
    if (df == 'four_week'):
        return df_all[df_all['Security term_4 WK'] == 1]
    if (df == 'eight_week'):
        return df_all[df_all['Security term_8 WK'] == 1]
    if (df == 'thirteen_week'):
        return df_all[df_all['Security term_13 WK'] == 1]
    if (df == 'seventeen_week'):
        return df_all[df_all['Security term_17 WK'] == 1]
    if (df == 'twenty_six_week'):
        return df_all[df_all['Security term_26 WK'] == 1]
    if (df == 'fifty_two_week'):
        return df_all[df_all['Security term_52 WK'] == 1]
    if (df == 'cmb'):
        return df_all[df_all['Security term_CMB'] == 1]
    return

def create_arrays(df):
    # Create variables to predict based on
    selected_columns = ['date', 'Total issue', '(SOMA) Federal Reserve banks', 'Depository institutions', 'Individuals', 'Dealers and brokers',
                        'Pension and Retirement funds and Ins. Co.', 'Investment funds', 'Foreign and international', 'Other and Noncomps', 'News Sentiment']
    X_array = np.array(df[selected_columns].values.tolist())
    Y_array = np.array(df['Auction high rate %'].values.tolist()).T
    return X_array, Y_array

def split_train_test(df, split_size, split_xy=True):
    ts = get_auction_type_df(df)

    # Split into a training set and a testing set
    train_size = int(len(ts) * split_size)
    train_ts, test_ts = ts[:train_size], ts[train_size:]

    # Splits training and testing sets into x and y
    x_train, y_train = create_arrays(train_ts)
    x_test, y_test = create_arrays(test_ts)

    if (split_xy):
        return x_train, y_train, x_test, y_test
    
    return train_ts, test_ts

def plot_data(df):
    df = get_auction_type_df(df)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    plt.plot(df.index, df['Auction high rate %'])
    plt.xlabel('Date')
    plt.ylabel('Auction high rate %')
    plt.show()

def create_data(df):
    X_array, Y_array = clean_data.create_arrays(df)
    final_array = np.column_stack((X_array, Y_array))

    dates = final_array[:, 0]
    difference = pd.Series(dates)-dates[0]

    df_diff = pd.DataFrame(difference) #find diff since original date
    df_diff['weeks'] = df_diff / pd.Timedelta(weeks=1) # convert to weeks
    weeks_array = pd.DataFrame(df_diff['weeks'])
    weeks_array = np.array(weeks_array.values.tolist()) # convert back to numpy

    final_array = np.delete(final_array, 0, axis=1) # remove the original datetime column

    # Concatenate along the second axis (i.e. columns)
    final_array = np.concatenate((weeks_array, final_array), axis=1)
    final_array = final_array[final_array[:, 0].argsort()]
    
    print(final_array)
    return final_array