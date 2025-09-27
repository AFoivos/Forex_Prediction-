import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexMomentumIndicators:
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
        Class for Momentum Indicators
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        open_col (str): Column name for open price
        high_col (str): Column name for high price
        low_col (str): Column name for low price
        close_col (str): Column name for close price
        volume_col (str): Column name for volume
        
        """
        
        print("="*50)
        print("MOMENTUM INDICATORS")
        print("="*50)
        print(" Available Fuctions \n1 add_rsi \n2 add_stochastic \n3 add_williams_r \n4 add_cci \n5 add_momentum \n6 get_all_momentum_indicators")
        print("="*50) 
        
        self.data = data.copy()
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        
        # Validate data_cols
        required_cols = [self.open_col, self.high_col, self.low_col, self.close_col]
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")
    
    def add_rsi(
        self, 
        periods: List[int] = [14, 21, 28],
    ):
        
        """
        Relative Strength Index
        
        Parameters:
        periods (List[int]): List of periods for RSI
        
        """
                
        for period in periods:
            col_name = f'momen_rsi_{period}'
            self.data[col_name] = talib.RSI(self.data[self.close_col], timeperiod=period)
            
            # RSI divergence
            self.data[f'{col_name}_trend'] = self.data[col_name].diff()
        
        return self.data
    
    def add_stochastic(
        self,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3,
        overbought: int = 80,
        oversold: int = 20
    ):
        
        """
        Stochastic Oscillator
        
        Parameters:
        fastk_period (int): Fast %K period
        slowk_period (int): Slow %K period
        slowd_period (int): Slow %D period
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        slowk, slowd = talib.STOCH(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            fastk_period = fastk_period,
            slowk_period = slowk_period,
            slowk_matype = 0,
            slowd_period = slowd_period,
            slowd_matype = 0
        )
        
        stochastic_columns = [
            'stoch_slowk', 'stoch_slowd', 'stoch_crossover',
            'stoch_overbought', 'stoch_oversold', 'stoch_signal'
        ]
        
        self.data['momen_stoch_slowk'] = slowk
        self.data['momen_stoch_slowd'] = slowd
                
        return self.data
    
    def add_williams_r(
        self,
        period: int = 14,
    ):
        
        """
        Williams %R
        
        Parameters:
        period (int): Period for Williams %R
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        willr = talib.WILLR(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod = period
        )
        
        willr_columns = [
            'williams_r', 'willr_overbought', 'willr_oversold',
            'willr_signal', 'willr_trend'
        ]
        
        self.data['momen_williams_r'] = willr
                
        return self.data
    
    def add_cci(
        self,
        period: int = 20,
    ):
        
        """
        Commodity Channel Index
        
        Parameters:
        period (int): Period for CCI
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        cci = talib.CCI(
            self.data[self.high_col],
            self.data[self.low_col],
            self.data[self.close_col],
            timeperiod = period
        )
        
        self.data['momen_cci'] = cci
                
        return self.data
    
    def add_momentum(
        self, 
        periods: List[int] = [10, 14, 20],
    ):
        
        """
        Momentum Indicator
        
        Parameters:
        periods (List[int]): List of periods for Momentum
        
        """
        
        for period in periods:
            col_name = f'momen_momentum_{period}'
            self.data[col_name] = talib.MOM(self.data[self.close_col], timeperiod=period)
        
        return self.data
    
    def get_all_momentum_indicators(
        self,
        rsi_periods: List[int] = [14, 21, 28],
        stochastic_fastk: int = 14,
        stochastic_slowk: int = 3,
        stochastic_slowd: int = 3,
        williams_period: int = 14,
        cci_period: int = 20,
        momentum_periods: List[int] = [10, 14, 20],
    ):
        
        """
        Adds all momentum indicators
        
        Parameters:
        rsi_periods (List[int]): List of periods for RSI
        stochastic_fastk (int): Fast %K period for Stochastic
        stochastic_slowk (int): Slow %K period for Stochastic
        stochastic_slowd (int): Slow %D period for Stochastic
        williams_period (int): Period for Williams %R
        cci_period (int): Period for CCI
        momentum_periods (List[int]): List of periods for Momentum
        
        """
        
        self.add_rsi(rsi_periods)
        self.add_stochastic(stochastic_fastk, stochastic_slowk, stochastic_slowd)
        self.add_williams_r(williams_period)
        self.add_cci(cci_period)
        self.add_momentum(momentum_periods)
        
        return self.data