import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

def clean_main_data():
    # read the 01-09 dataset
    df_09 = pd.read_excel('Aug2001-Sep2009.xls')
    # read the 09-2023 dataset
    df_23 = pd.read_excel('Oct2009-May8,2023.xls')
    # read sentiment dataset
    df_news_sentiment = pd.read_excel('econ_news_sentiment.xlsx', 'Data')
    # read effective federal funds rate dataset
    df_effr = pd.read_csv('effectiveffr.csv')

    # define the columns
    df_09.columns = df_09.iloc[2]
    df_23.columns = df_23.iloc[2]

    df_09 = df_09.dropna()
    df_23 = df_23.dropna()

    # concatenate the datasets
    df_all = pd.concat([df_09, df_23], ignore_index=True)

    # clean the data, dropping the relevant variables/rows and standardizing between datasets, incl. with regexes
    df_all = df_all.drop([0])
    df_all.drop(df_all.tail(2).index, inplace = True)
    df_all = df_all.drop('Cusip', axis=1)
    df_all = df_all.dropna()
    df_all['Security term'] = df_all['Security term'].str.replace('13-Week Bill', '13 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('4-Week Bill', '4 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('26-Week Bill', '26 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('52-Week Bill', '52 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('8-Week Bill', '8 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('17-Week Bill', '17 WK')
    df_all['Security term'] = df_all['Security term'].str.replace('CASH', 'CMB')
    df_all['Security term'] = df_all['Security term'].str.replace('.*CMB*.', 'CMB', regex=True)
    df_all = df_all.drop(index=df_all[df_all['Security term'].isin(['Security term'])].index)

    # Convert categorical variables of security term to dummies
    df_all = pd.get_dummies(df_all, columns=['Security term'])

    # merge the dataframes on the date column
    df_all = df_all.rename({'Issue date': 'date'}, axis=1)
    df_all['date'] = pd.to_datetime(df_all['date'])

    df_effr = df_effr.rename({'Effective Date': 'date'}, axis=1)
    df_effr = df_effr.dropna()
    df_effr['date'] = pd.to_datetime(df_all['date'])

    df_all = pd.merge(df_all, df_news_sentiment, on='date', how='inner')

    # sort them in time order
    df_all = df_all.sort_values(by='date')

    return df_all

print(clean_main_data().columns)
# clean_main_data().to_excel('cleaned_data.xlsx', index=False)