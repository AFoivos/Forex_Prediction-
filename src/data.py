import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def get_data(file_path, symbole) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.

    Args:
        file_path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    column_names = [ 'Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Something_Else']
    df = pd.read_csv(file_path, 
                     names=column_names, 
                     header=None, 
                     sep=',',
                     encoding='utf-16',
                     parse_dates=['Time'],
                    )
    
    df = df[['Close','Time' ]]
    df['Time'] = pd.to_datetime(df['Time'])
    df.set_index('Time', inplace=True)
    
    print (f'First 5 rows:\n{df.head()} \n')
    print (f'Last 5 rows:\n{df.tail()} \n')
    print (f'Shape: {df.shape} \n')
    print (f'Info: {df.info()} \n')
    print (f'Null values:\n{df.isnull().sum()} \n')
    print (f'Duplicated values: {df.duplicated().sum()}\n')    
    print (f'Data types:\n{df.dtypes}\n')
    print (f'Statistics:\n{df.describe()}\n')
    
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Close Price', color='blue')
    plt.title( symbole +' Close Price Over Time')
    plt.xlabel('Time')
    plt.ylabel('Close Price')
    plt.legend()
    plt.grid()
    plt.show()
    
    return df