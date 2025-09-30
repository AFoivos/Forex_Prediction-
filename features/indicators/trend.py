import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexTrendIndicators:
    def __init__(
        self, 
        data: pd.DataFrame,
        open_col: str = 'open',
        high_col: str = 'high', 
        low_col: str = 'low', 
        close_col: str = 'close',
        volume_col: str = 'volume',
    ):
        
        """
        Class for Trend Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("TREND INDICATORS")
        print("="*50)
        print(" Available Fuctions: \n1 add_sma \n2 add_ema \n3 add_macd \n4 add_adx \n5 add_parabolic_sar \n6 generate_all_trend_indicators")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        self.parameters = {}
        
        self.trend_data = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self._validate_columns()
    
    def _validate_columns(
        self, 
        columns: list[str] = None,
    ):  
        
        """
        Validate that required indicator columns exist
        
        Parameters:
        columns (list[str]): List of column names to validate
            
        """
        
        required_cols = [
            self.close_col,
            self.high_col,
            self.low_col,
            self.open_col,
            self.volume_col
        ]
        
        if columns is not None:
            required_cols.extend(columns)
        
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    def _is_nested_list(
        self, 
        lst
    ):
        
        """
        Check if a list is nested or not
        
        Parameters:
        lst (list): List to check
        
        """
        
        if not all(isinstance(item, list) for item in lst):
            return [lst]
        else:
            return lst
        
    def add_sma(
        self, 
        periods: List[int] = [10, 20, 50, 100, 200]
    ):
        
        """
        Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for SMA
        
        """ 
        
        self.parameters['sma_params'] = periods
        
        for period in periods:
            col_name = f"sma_{period}"
            
            # SMA
            self.trend_data[col_name] = talib.SMA(
                self.data[self.close_col], 
                timeperiod=period
            )
            
            # SMA slope
            self.trend_data[f"{col_name}_slope"] = self.trend_data[col_name].diff()
        
        return self.trend_data
        
    def add_ema(
        self, 
        periods: List[int] = [10, 20, 50, 100, 200]
    ):
        
        """
        Exponential Moving Averages 
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        self.parameters['ema_params'] = periods
        
        for period in periods:
            col_name = f'ema_{period}'
            
            # EMA
            self.trend_data[col_name] = talib.EMA(
                self.data[self.close_col], 
                timeperiod=period
            )
        
            # EMA slope
            self.trend_data[f'{col_name}_slope'] = self.trend_data[col_name].diff()

        return self.trend_data
        
    def add_macd(
        self, 
        fast_slow_signal_periods: List[int] = [12, 26, 9],
    ):
        
        """
        MACD Indicator 
        
        Parameters:
        fast_slow_signal_periods (List[int]): List nested or not, [fastperiod, slowperiod, signalperiod] for MACD
        
        """
        
        self.parameters['macd_params'] = fast_slow_signal_periods
        
        fast_slow_signal_periods = self._is_nested_list(fast_slow_signal_periods)
        
        for lst in fast_slow_signal_periods:
            fast = lst[0]
            slow = lst[1]
            signal = lst[2]
            
            macd, macd_signal, macd_hist = talib.MACD(
            self.data[self.close_col],
            fastperiod=fast,
            slowperiod=slow,
            signalperiod=lst[2]
        )

            # MACD
            self.trend_data[f'macd_{fast}'] = macd
            self.trend_data[f'macd_signal_{signal}'] = macd_signal
            self.trend_data[f'macd_hist_fast_{fast}_slow{slow}_sig{signal}'] = macd_hist
            
            # MACD slopes
            self.trend_data[f"macd_{fast}_slope"] = macd.diff()
            self.trend_data[f"macd_signal_{signal}_signal_slope"] = macd_signal.diff()
            
        return self.trend_data
    
    def add_adx(
        self,
        periods: List[int] = [14, 21, 28],
    ):

        """
        Average Directional Movement Index
        
        Parameters:
        periods (List[int]): List of periods for ADX
        
        """  
        
        self.parameters['adx_params'] = periods
       
        for period in periods:
            # Average Directional Movement Index
            adx = talib.ADX(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod=period
            )
            
            # Positive Directional 
            plus_di = talib.PLUS_DI(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod=period
            )
            
            # Negative Directional 
            minus_di = talib.MINUS_DI(
                self.data[self.high_col],
                self.data[self.low_col],
                self.data[self.close_col],
                timeperiod=period
            )
            
            col_name = f'adx_{period}'
            
            # ADX
            self.trend_data[col_name] = adx
            self.trend_data[f'{col_name}_plus'] = plus_di
            self.trend_data[f'{col_name}_minus'] = minus_di
            
            # ADX Slope
            self.trend_data[f'{col_name}_slope'] = adx.diff()
            self.trend_data[f'{col_name}_plus_di_slope'] = plus_di.diff()
            self.trend_data[f'{col_name}_minus_di_slope'] = minus_di.diff()
                
        return self.trend_data
        
    def add_parabolic_sar(
        self, 
        acc_max: List[float] = [0.02, 0.2],
    ):
        
        """
        Parabolic Stop and Reverse
        
        Parameters:
        acc_max (List[int]): List nested or not, [acceleration, maximum] for SAR
        
        """
        
        self.parameters['sar_params'] = acc_max
        
        acc_max = self._is_nested_list(acc_max)
        
        for lst in acc_max:
            sar = talib.SAR(
                self.data[self.high_col],
                self.data[self.low_col],
                acceleration=lst[0],
                maximum=lst[1]
            )
            col_name = f'sar_{lst[0]}_{lst[1]}'

            # Parabolic sar
            self.trend_data[col_name] = sar
            
            # Parabolic sar slope
            self.trend_data[f"{col_name}_slope"] = self.trend_data[col_name].diff()
                   
        return self.trend_data     
         
    def generate_all_trend_indicators(
        self,
        sma_periods: List[int] = [10, 20, 50, 100, 200],
        ema_periods: List[int] = [10, 20, 50, 100, 200],
        macd: List[int] = [12, 26, 9],
        adx_periods: List[int] = [14, 21, 28],
        parabolic_sar: List[int] = [0.02, 0.2],
    ):
        
        """
        Adds all trend indicators
        
        Parameters:
        sma_periods (List[int]): List of periods for SMA
        ema_periods (List[int]): List of periods for EMA
        macd (List[int]): List nested or not, [fastperiod, slowperiod, signalperiod] for MACD
        adx_periods (List[int]): List of periods for ADX
        parabolic_sar ema_periods (List[int]): List of nested or not, [acceleration, maximum] for SAR
        
        """
        
        self.add_sma(periods = sma_periods)
        self.add_ema(periods = ema_periods)
        self.add_macd(fast_slow_signal_periods = macd)
        self.add_adx(periods = adx_periods)
        self.add_parabolic_sar(acc_max = parabolic_sar)
        
        count_removed_rows = self.data.shape[0] - self.trend_data.shape[0]
    
        print('='*50)
        print('Data Info')
        print(self.trend_data.info())
        print('='*50)
        print(f'Shape of data {self.trend_data.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.trend_data, self.parameters
        