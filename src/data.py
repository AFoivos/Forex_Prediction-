import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------   
#----------------------------------------------------------------------------------------------------------------------------------------   

def first_look(file_path, symbole) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    !!!!!! THE FILE MUST BE FETCHED FROM MT5 !!!!!!

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    new_column_names = ['date', 'time', 'open', 'high', 'low', 'close', 'tickvol', 'vol', 'spread']

    df = pd.read_csv(file_path, 
                    header=None,   
                    sep='\t',
                    names=new_column_names, 
                    skiprows=1  
                    )
    
    df['time'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    
    df.drop(columns=['date', 
                     'tickvol', 
                     'vol', 
                     'spread'], inplace=True)
    
    df = df[df['time'].dt.year >= 2020]
    
    df.set_index('time', inplace=True)
    
    
    print (f'First 5 rows:\n{df.head()} \n')
    print('---'*40)
    print (f'Last 5 rows:\n{df.tail()} \n')
    print('---'*40)
    print (f'Shape: {df.shape} \n')
    print('---'*40)
    print (f'Info: {df.info()} \n')
    print('---'*40)
    print (f'Null values:\n{df.isnull().sum()} \n')
    print('---'*40)
    print (f'Duplicated values: {df.duplicated().sum()}\n')    
    print('---'*40)
    print (f'Data types:\n{df.dtypes}\n')
    print('---'*40)
    print (f'Statistics:\n{df.describe()}\n')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['close'], label='Close Price', color='blue')
    plt.title( symbole +' Close Price Over Time')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()
    
    return df
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------

def pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'PercentageChange' columns to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame .

    Returns:
    pd.DataFrame: DataFrame with added 'PercentageChange' features.
    """
    new_column_names = []
    for col in df.columns:
        new_col_name = f'{col}_pct_change'
        df[new_col_name] = df[col].pct_change()
        new_column_names.append(new_col_name)
    
    sum_of_null = df.isnull().sum().sum()    
    df = df.dropna()
    df = df[new_column_names]
    
    print (f'First 3 rows:\n{df.head(3)} \n')
    print ('---'*40)
    print (f'Last 3 rows:\n{df.tail(3)} \n')
    print ('---'*40)
    print (f'Statistics:\n{df.describe()} \n')
    print ('---'*40)
    print (f'Info: {df.info()} \n')
    print ('---'*40)
    print (f'Null values before dropping:\n{sum_of_null}, \n')
    print ('---'*40)
    print (f'Shape: {df.shape} \n')
    print ('---'*40)
    print ('Columns names:\n', df.columns, '\n')
    
    return df 