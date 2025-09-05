import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

def pct_and_lags(df: pd.DataFrame, num_lags: int = 10) -> pd.DataFrame:
    """
    Adds a 'PercentageChange' column and lagged features to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame with a 'Close' column.

    Returns:
    pd.DataFrame: DataFrame with added 'PercentageChange' and lagged features.
    """
    df['PercentageChange'] = df['Close'].pct_change().round(4)
    
    num = num_lags + 1
      
    for i in range(1, num):
        df[f'Close_Lag_{i}'] = df['PercentageChange'].shift(i).round(4)
        
    df = df.dropna()
    
    print (f'First 5 rows:\n{df.head()} \n')
    print('---'*20)
    print (f'Last 5 rows:\n{df.tail()} \n')
    print('---'*20)
    print (f'Shape: {df.shape} \n')
    print('---'*20)
    print (f'Info: {df.info()} \n')
    print('---'*20)
    print (f'Null values:\n{df.isnull().sum()} \n')
    print('---'*20)
    print (f'Duplicated values: {df.duplicated().sum()}\n')    
    print('---'*20)
    print (f'Data types:\n{df.dtypes}\n')
    print('---'*20)
    print (f'Statistics:\n{df.describe()}\n')

    last_500 = df.tail(500)

    plt.figure(figsize=(12, 6))
    plt.plot(last_500.index, last_500['PercentageChange'], label='Close Price', color='blue')
    plt.title('Percentage Change Over Time')
    plt.xlabel('Time')
    plt.ylabel('Percentage Change')
    plt.legend()
    plt.grid()
    plt.show()
    
    last_24_hours = df.tail(24)
    plt.figure(figsize=(12, 6))
    for i in range(1, num):
        plt.plot(last_24_hours .index, last_24_hours [f'Close_Lag_{i}'], label=f'Close Lag {i}', linestyle='--')
    plt.plot(last_24_hours.index, last_24_hours['PercentageChange'], label='Original Close Price', color='black', linewidth=2)

    plt.title('Percentage Change vs. Lags', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Percentage Change', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    correlation_matrix = df.drop(columns='Close').corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap of Percentage Change and Lags', fontsize=16)
    plt.show()

    return df