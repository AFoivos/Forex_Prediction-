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
        print(" Available Fuctions: \n1 add_sma \n2 add_ema \n3 add_macd \n4 add_adx \n5 add_parabolic_sar")
        print("="*50)
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self._validate_columns()
    
    def _validate_columns(self):
        
        #Validate data_cols
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
        
    def add_sma(
        self, 
        periods: List[int] = [10, 20, 50, 100, 200]
    ):
        
        """
        Simple Moving Averages
        
        Parameters:
        periods (List[int]): List of periods for SMA
        
        """ 
        
        for period in periods:
            col_name = f'trend_sma_{period}'
            self.data[col_name] = talib.SMA(self.data[self.close_col], timeperiod=period)
            
            # Slope of SMA
            self.data[f'{col_name}_slope'] = self.data[col_name].diff()
        
        return self.data
        
    def add_ema(
        self, 
        periods: List[int] = [10, 20, 50, 100, 200]
    ):
        
        """
        Exponential Moving Averages 
        
        Parameters:
        periods (List[int]): List of periods for EMA
        
        """
        
        for period in periods:
            col_name = f'trend_ema_{period}'
            self.data[col_name] = talib.EMA(self.data[self.close_col], timeperiod=period)
        
            # EMA slope
            self.data[f'{col_name}_slope'] = self.data[col_name].diff()

        return self.data
        
    def add_macd(
        self, 
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ):
        
        """
        MACD Indicator 
        
        Parameters:
        fastperiod (int): Fast period for MACD
        slowperiod (int): Slow period for MACD
        signalperiod (int): Signal period for MACD
        
        """
        
        macd, macd_signal, macd_hist = talib.MACD(
            self.data[self.close_col],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        
        self.data['trend_macd'] = macd
        self.data['trend_macd_signal'] = macd_signal
        self.data['trend_macd_hist'] = macd_hist
        
        return self.data
    
    def add_adx(
        self,
        period: int = 14,
    ):

        """
        Average Directional Movement Index
        
        Parameters:
        period (int): Period for ADX
        
        """  

        adx = talib.ADX(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        # Positive and Negative Directional Indicators
        plus_di = talib.PLUS_DI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        minus_di = talib.MINUS_DI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod=period
        )
        
        self.data['trend_adx'] = adx
        self.data['trend_plus_di'] = plus_di
        self.data['trend_minus_di'] = minus_di
        
        # ADX Slope
        self.data['trend_adx_slope'] = self.data['trend_adx'].diff()
                
        return self.data
        
    def add_parabolic_sar(
        self, 
        acceleration: float = 0.02,
        maximum: float = 0.2
    ):
        
        """
        Parabolic Stop and Reverse
        
        Parameters:
        acceleration (float): Acceleration parameter for SAR
        maximum (float): Maximum parameter for SAR
        
        """
        
        sar = talib.SAR(
            self.data[self.high_col],
            self.data[self.low_col],
            acceleration=acceleration,
            maximum=maximum
        )
        
        self.data['trend_parabolic_sar'] = sar
        self.data['trend_parabolic_sar_slope'] = self.data['trend_parabolic_sar'].diff()
                   
        return self.data     
         
    def get_all_trend_indicators(
        self,
        sma_periods: List[int] = [10, 20, 50, 100, 200],
        ema_periods: List[int] = [10, 20, 50, 100, 200],
        macd_fastperiod: int = 12,
        macd_slowperiod: int = 26,
        macd_signalperiod: int = 9,
        adx_period: int = 14,
        parabolic_sar_acceleration: float = 0.02,
        parabolic_sar_maximum: float = 0.2,
    ):
        
        """
        Adds all trend indicators
        
        Parameters:
        sma_periods (List[int]): List of periods for SMA
        ema_periods (List[int]): List of periods for EMA
        macd_fastperiod (int): Fast period for MACD
        macd_slowperiod (int): Slow period for MACD
        macd_signalperiod (int): Signal period for MAC  
        adx_period (int): Period for ADX
        parabolic_sar_acceleration (float): Acceleration parameter for SAR
        parabolic_sar_maximum (float): Maximum parameter for SAR        
        
        """
        
        self.add_sma(sma_periods)
        self.add_ema(ema_periods)
        self.add_macd(macd_fastperiod, macd_slowperiod, macd_signalperiod)
        self.add_adx(adx_period)
        self.add_parabolic_sar(parabolic_sar_acceleration, parabolic_sar_maximum)
        
        return self.data
    