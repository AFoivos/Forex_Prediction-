import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns  

def lags_and_rolls(df: pd.DataFrame, num_lags: int = 10, num_rolls: int = 10) -> pd.DataFrame:
    """
    Adds lagged and rolling features to the DataFrame.   
    """   
    num_l = num_lags + 1
    num_r_std = num_rolls + 2
    num_r_mean = num_rolls + 1
    
    for i in range(1, num_l):
        df[f'close_lag_{i}'] = df['close'].shift(i)
        
    for i in range(2, num_r_std):
        df[f'close_roll_std{i}'] = df['close'].rolling(i).std()
        
    for i in range(1, num_r_mean):
        df[f'close_roll_mean{i}'] = df['close'].rolling(i).mean()
        
   
    #df = df.dropna().copy()
    
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
    print (f'Duplicated values: {df.duplicated().sum()} \n')    
    print('---'*40)
    print (f'Data types:\n{df.dtypes} \n')
    print('---'*40)
    print (f'Statistics:\n{df.describe()} \n')

    last_24_hours = df.tail(24)

    plt.figure(figsize=(12, 6))
    for i in range(1, num_l):
        plt.plot(last_24_hours.index, last_24_hours [f'close_lag_{i}'], label=f'Close Lag {i}', linestyle='--')
        
    plt.title('Close Lags', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value Change', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for i in range(1, num_r_mean):
        plt.plot(last_24_hours.index, last_24_hours [f'close_roll_mean{i}'], label=f'Close Roll Mean {i}', linestyle='--')
        
    plt.title('Rolling Means', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value Change', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()
    
    plt.figure(figsize=(12, 6))
    for i in range(2, num_r_std):
        plt.plot(last_24_hours.index, last_24_hours [f'close_roll_std{i}'], label=f'Close Roll Std {i}', linestyle='--')
        
    plt.title('Rolling Stds', fontsize=16)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Value Change', fontsize=12)
    plt.legend()
    plt.grid(True)
    plt.show()  

    
    return df
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------------------------------------------------

def indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds technical indicators to the DataFrame.

    Parameters:
    df (pd.DataFrame): Input DataFrame.

    Returns:
    pd.DataFrame: DataFrame with added technical indicators.
    """
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.volume import OnBalanceVolumeIndicator, ChaikinMoneyFlowIndicator
    from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    import matplotlib.pyplot as plt
    import numpy as np  
    

    # Relative Strength Index (RSI)
    rsi_indicator = RSIIndicator(close=df['close'], window=14)
    df['RSI_14'] = rsi_indicator.rsi()

    # Simple Moving Average (SMA)
    sma_indicator = SMAIndicator(close=df['close'], window=14)
    df['SMA_14'] = sma_indicator.sma_indicator()

    # Exponential Moving Average (EMA)
    ema_indicator = EMAIndicator(close=df['close'], window=14)
    df['EMA_14'] = ema_indicator.ema_indicator()
    
    # Stochastic Oscillator
    df['14_Low'] = df['low'].rolling(window=14).min()
    df['14_High'] = df['high'].rolling(window=14).max()
    df['%K'] = 100 * ((df['close'] - df['14_Low']) / (df['14_High'] - df['14_Low']))
    df['%D'] = df['%K'].rolling(window=3).mean()
    df.drop(['14_Low', '14_High'], axis=1, inplace=True)
    
    
    # Average True Range (ATR)
    df['ATR'] = df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min()
    df['ATR'] = df['ATR'].rolling(window=14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = np.where(df['close'] > df['close'].shift(1), df['volume'], np.where(df['close'] < df['close'].shift(1), -df['volume'], 0))
    df['OBV'] = df['OBV'].cumsum()      
    
    # Commodity Channel Index (CCI)
    df['Middle_Line'] = df['close'].rolling(window=20).mean()           
    df['Mean_Deviation'] = df['close'].rolling(window=20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    df['CCI'] = (df['close'] - df['Middle_Line']) / (0.015 * df['Mean_Deviation'])
    df.drop(['Middle_Line', 'Mean_Deviation'], axis=1, inplace=True)
    
    # Moving Average Convergence Divergence (MACD)
    ema_12 = EMAIndicator(close=df['close'], window=12).ema_indicator()
    ema_26 = EMAIndicator(close=df['close'], window=26).ema_indicator()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].rolling(window=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']    
    
    # True Range (TR)
    df['TR'] = df[['high', 'low', 'close']].max(axis=1) - df[['high', 'low', 'close']].min(axis=1)
    
    # Bollinger Bands
    bollinger = BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_Middle'] = bollinger.bollinger_mavg()
    df['Bollinger_Upper'] = bollinger.bollinger_hband()
    df['Bollinger_Lower'] = bollinger.bollinger_lband()

    df = df.dropna().copy()
    
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
    print (f'Duplicated values: {df.duplicated().sum()} \n')    
    print('---'*40)
    print (f'Data types:\n{df.dtypes} \n')
    print('---'*40)
    print (f'Statistics:\n{df.describe()} \n')

    # Heatmap of correlations
    plt.figure(figsize=(14, 10))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', cbar=True)
    plt.title('Feature Correlation Heatmap', fontsize=16)
    plt.show()
    
    return df