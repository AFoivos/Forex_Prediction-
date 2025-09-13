import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import talib

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)
    
import src.data_handle as dt
#file_path = os.path.join(project_root, 'src', 'data_handle', 'load_clean_first_look.py')
from src.data_handle.first_look_and_clean import FirstLook
from src.data_handle.eda import ForexEDA

import warnings
warnings.filterwarnings('ignore')

class TechnicalIndicators:
    def __init__(self, df:pd.DataFrame,
                 printer:bool = True,
                 calulate_all:bool = False
                 ):
        
        """
        Initialize Technical Indicators calculator
        
        Parameters:
        df: pandas DataFrame 
        printer: bool, if True prints data load confirmation  
        
        """
        
        self.data = df.copy()
        
        print('Data loaded successfully!')
        print(f'Shape: {self.data.shape}')
        print("\n" + "="*50)
                
        self.indicators = {}
        self.signals = {}
        
        if calulate_all == True:
            self.calculate_all_indicators()
    
    def calculate_moving_averages(self, 
                                  windows:list = [5, 10, 20, 50, 200]
                                  ):
       
        """
        Calculate Moving Averages using TA-Lib
        Parameters:
        windows: list of moving average windows
        
        """
        print("CALCULATING MOVING AVERAGES WITH TA-LIB")
        print("=" * 50)
        
        close_prices = self.data['close'].values
        
        for window in windows:
            # SMA
            sma_col = f'SMA_{window}'
            self.data[sma_col] = talib.SMA(close_prices, timeperiod=window)
            
            # EMA
            ema_col = f'EMA_{window}'
            self.data[ema_col] = talib.EMA(close_prices, timeperiod=window)
            
            self.indicators[sma_col] = self.data[sma_col]
            self.indicators[ema_col] = self.data[ema_col]
            
            print(f" Calculated {sma_col} and {ema_col}")
            
            self.indicators[sma_col] = self.data[sma_col]
            self.indicators[ema_col] = self.data[ema_col]
        
        return self.data
    
    def calculate_rsi(self, 
                      window:int = 14
                      ):
        
        """
        Calculate RSI using TA-Lib
        Parameters:
        window: time period for RSI calculation
        """
        print(f"\nCALCULATING RSI ({window}) WITH TA-LIB")
        print("=" * 50)
        
        if 'close' not in self.data.columns:
            print("Error: 'close' column not found!")
            return
        
        close_prices = self.data['close'].values
        
        rsi = talib.RSI(close_prices, timeperiod=window)

        self.data[f'RSI_{window}'] = rsi
        self.indicators[f'RSI_{window}'] = rsi
        
        
        print(f" RSI_{window} calculated")
                
        return rsi
    
    def calculate_macd(self,
                       fast:int = 12,
                       slow:int = 26,
                       signal:int = 9
                       ):
        
        """
        Calculate MACD using TA-Lib
        Parameters:
        fast: fast moving average period
        slow: slow moving average period
        signal: signal line period
        """
        print(f"\nCALCULATING MACD ({fast}, {slow}, {signal}) WITH TA-LIB")
        print("=" * 50)
        
        close_prices = self.data['close'].values

        macd, signal_line, histogram = talib.MACD(close_prices, 
                                                fastperiod=fast, 
                                                slowperiod=slow, 
                                                signalperiod=signal
                                                )
        
        self.data[f'MACD_{fast}_{slow}_{signal}'] = macd
        self.data[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
        self.data[f'MACD_Histogram_{fast}_{slow}_{signal}'] = histogram
        
        self.indicators[f'MACD_{fast}_{slow}_{signal}'] = macd
        self.indicators[f'MACD_Signal_{fast}_{slow}_{signal}'] = signal_line
        self.indicators[f'MACD_Histogram_{fast}_{slow}_{signal}'] = histogram
        
        print(" MACD, Signal Line, and Histogram calculated")
        return macd, signal_line, histogram
    
    def calculate_bollinger_bands(self,
                                  window:int = 20,
                                  num_std:int = 2
                                  ):

        """
        Calculate Bollinger Bands using TA-Lib
        Parameters:
        window: time period for BB calculation
        num_std: number of standard deviations for BB
        """
        
        print(f"\nCALCULATING BOLLINGER BANDS ({window}, {num_std}σ) WITH TA-LIB")
        print("=" * 50)
        
        if 'close' not in self.data.columns:
            print("Error: 'close' column not found!")
            return
        
        close_prices = self.data['close'].values
        
        upper, middle, lower = talib.BBANDS(close_prices, 
                                            timeperiod=window, 
                                            nbdevup=num_std, 
                                            nbdevdn=num_std)

        
        self.data[f'BB_Upper_{window}_{num_std}'] = upper
        self.data[f'BB_Middle_{window}_{num_std}'] = middle
        self.data[f'BB_Lower_{window}_{num_std}'] = lower
        self.data[f'BB_Width_{window}_{num_std}'] = (upper - lower) / middle
        
        self.indicators[f'BB_Upper_{window}_{num_std}'] = upper
        self.indicators[f'BB_Middle_{window}_{num_std}'] = middle
        self.indicators[f'BB_Lower_{window}_{num_std}'] = lower
        self.indicators[f'BB_Width_{window}_{num_std}'] = (upper - lower) / middle
        
        print(" Bollinger Bands calculated")
        return upper, middle, lower
    
    def calculate_stochastic_oscillator(self, 
                                        k_window:int = 14,
                                        d_window:int = 3
                                        ):
        
        """
        Calculate Stochastic Oscillator using TA-Lib
        Parameters:
        k_window: time period for %K calculation
        d_window: time period for %D calculation
        """
        print(f"\nCALCULATING STOCHASTIC OSCILLATOR ({k_window}, {d_window}) WITH TA-LIB")
        print("=" * 50)
        
        for col in ['high', 'low', 'close']:
            if col not in self.data.columns:
                print(f"Error: '{col}' column not found!")
                return
        
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        
        slowk, slowd = talib.STOCH(high, low, close, 
                                    fastk_period=k_window, 
                                    slowk_period=d_window, 
                                    slowk_matype=0, 
                                    slowd_period=d_window, 
                                    slowd_matype=0)
        
        self.data[f'Stoch_%K_{k_window}_{d_window}'] = slowk
        self.data[f'Stoch_%D_{k_window}_{d_window}'] = slowd
        
        self.indicators[f'Stoch_%K_{k_window}_{d_window}'] = slowk
        self.indicators[f'Stoch_%D_{k_window}_{d_window}'] = slowd
        
        print(" Stochastic Oscillator calculated")
        return slowk, slowd
    
    def calculate_atr(self, 
                      window:int = 14
                      ): 
        
        """
        Calculate ATR using TA-Lib
        Parameters:
        window: time period for ATR calculation
        """
        print(f"\nCALCULATING AVERAGE TRUE RANGE ({window}) WITH TA-LIB")
        print("=" * 50)
        
        for col in ['high', 'low', 'close']:
            if col not in self.data.columns:
                print(f"Error: '{col}' column not found!")
                return
        
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        
        atr = talib.ATR(high, low, close, timeperiod=window)
     
        self.data[f'ATR_{window}'] = atr
        self.indicators[f'ATR_{window}'] = atr
        
        print(" Average True Range calculated")
        return atr
    
    def calculate_adx(self,
                      window:int = 14
                      ):
        
        """
        Calculate ADX using TA-Lib
        
        Parameters:
        window: time period for ADX calculation
        """
        print(f"\nCALCULATING ADX ({window}) WITH TA-LIB")
        print("=" * 50)
        
        for col in ['high', 'low', 'close']:
            if col not in self.data.columns:
                print(f"Error: '{col}' column not found!")
                return
        
        high = self.data['high'].values
        low = self.data['low'].values
        close = self.data['close'].values
        
        adx = talib.ADX(high, low, close, timeperiod=window)

        self.data[f'ADX_{window}'] = adx
        self.indicators[f'ADX_{window}''ADX'] = adx
        
        print(" ADX calculated")
        return adx
    
    def calculate_obv(self):
        """
        Calculate On-Balance Volume using TA-Lib
        """
        if 'volume' in self.data.columns:
            print(f"\nCALCULATING ON-BALANCE VOLUME WITH TA-LIB")
            print("=" * 50)
            
            if 'close' not in self.data.columns:
                print("Error: 'close' column not found!")
                return
            
            close = self.data['close'].values
            volume = self.data['volume'].values if 'volume' in self.data.columns else np.ones(len(close))
            
            obv = talib.OBV(close, volume)

            self.data['OBV'] = obv
            self.indicators['OBV'] = obv
            
            print(" On-Balance Volume calculated")
            return obv
        else:
            print("Error: 'volume' column not found!")
    
    def calculate_lags(self, 
                       lags:int = 5,
                       ):
        
        """
        Create lagged features
        Parameters:
        lags: number of lags to create
        """
        print(f"\nCALCULATING LAGGED FEATURES ({lags} lags)")
        print("=" * 50)
        
        for lag in range(1, lags + 1):
            lag_col = f'Close_Lag_{lag}'
            self.data[lag_col] = self.data['close'].shift(lag)
            self.indicators[lag_col] = self.data[lag_col]
            print(f" Created {lag_col}")
        
        returns = self.data['close'].pct_change()
        for lag in range(1, lags + 1):
            ret_lag_col = f'Return_Lag_{lag}'
            self.data[ret_lag_col] = returns.shift(lag)
            self.indicators[ret_lag_col] = self.data[ret_lag_col]
            
            print(f" Created {ret_lag_col}")
    
    def calculate_all_indicators(self,
                                 mas:list = [5, 10, 20, 50, 200],
                                 rsi:int = 14,
                                 macd:tuple = (12,26,9),
                                 bb:tuple = (20,2),
                                 stoch:tuple = (14,3),
                                 atr:int = 14,
                                 adx:int = 14,
                                 lags:int = 5):
        """
        Calculate all technical indicators using TA-Lib
        Parameters:
        mas: list of moving average windows
        rsi: time period for RSI calculation
        macd: tuple of (fast, slow, signal) periods for MACD
        bb: tuple of (window, num_std) for Bollinger Bands
        stoch: tuple of (k_window, d_window) for Stochastic Oscillator
        atr: time period for ATR calculation
        adx: time period for ADX calculation
        lags: number of lags to create
    
        """
        print("CALCULATING ALL TECHNICAL INDICATORS WITH TA-LIB")
        print("=" * 60)
        
        self.calculate_moving_averages(mas)
        self.calculate_rsi(rsi)
        self.calculate_macd(*macd)
        self.calculate_bollinger_bands(*bb)
        self.calculate_stochastic_oscillator(*stoch)
        self.calculate_atr(atr)
        self.calculate_adx(adx)
        self.calculate_lags(lags)
        
        if 'volume' in self.data.columns:
            self.calculate_obv()