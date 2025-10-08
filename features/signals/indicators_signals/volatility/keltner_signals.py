import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexKeltnerSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
    ):
        
        """
        Class for Keltner Channels signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("KELTNER CHANNELS SIGNAL GENERATION")
            print("="*50)
            print("Available functions: \n1 keltner_price_position_signals \n2 keltner_breakout_signals \n3 keltner_squeeze_signals \n4 keltner_trend_signals \n5 keltner_divergence_signals \n6 generate_all_keltner_signals")
            print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index = self.data.index
        )
        
        self.keltner_upper = []
        self.keltner_middle = []
        self.keltner_lower = []
        
        self.parameters = [20, 10, 2.0] if parameters is None else parameters
        
        self._validate_columns()
        self._extract_column_names(parameters = parameters)
        
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
        
        return all(isinstance(item, list) for item in lst)
    
    def _extract_column_names(
        self,
        parameters: List = None,
    ):
        
        """
        Extract Keltner Channels column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                ema_period = sublist[0]
                atr_period = sublist[1]
                multiplier = sublist[2]
                
                col_name = f'keltner_ema_{ema_period}_atr_{atr_period}_{multiplier}'
                
                self.keltner_upper.extend([f'{col_name}_upper'])
                self.keltner_middle.extend([f'{col_name}_middle'])
                self.keltner_lower.extend([f'{col_name}_lower'])
        else:
            ema_period = parameters[0]
            atr_period = parameters[1]
            multiplier = parameters[2]
            
            col_name = f'keltner_ema_{ema_period}_atr_{atr_period}_{multiplier}'
            
            self.keltner_upper.extend([f'{col_name}_upper'])
            self.keltner_middle.extend([f'{col_name}_middle'])
            self.keltner_lower.extend([f'{col_name}_lower'])
    
    def keltner_price_position_signals(
        self
    ):
        
        """
        Keltner Channels Price Position Signals
        2 = Price above Upper Channel (Strong Uptrend)
        1 = Price below Lower Channel (Strong Downtrend)
        0 = Price within Channels (Normal)
        
        """
        
        for upper, lower in zip(self.keltner_upper, self.keltner_lower):
            self._validate_columns(columns = [upper, lower])
            
            above_upper = self.data[self.close_col] > self.data[upper]
            below_lower = self.data[self.close_col] < self.data[lower]
            
            self.signals[f'{upper}_{lower}_position'] = np.select(
                [above_upper, below_lower],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def keltner_breakout_signals(
        self
    ):
        
        """
        Keltner Channels Breakout Signals
        2 = Breakout above Upper Channel
        1 = Breakdown below Lower Channel
        0 = No breakout
        
        """
        
        for upper, lower in zip(self.keltner_upper, self.keltner_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            breakout_above = (
                (self.data[self.close_col] > self.data[upper]) & 
                (self.data[self.close_col].shift(1) <= self.data[upper].shift(1))
            )
            
            breakdown_below = (
                (self.data[self.close_col] < self.data[lower]) & 
                (self.data[self.close_col].shift(1) >= self.data[lower].shift(1))
            )
            
            self.signals[f'{upper}_{lower}_breakout'] = np.select(
                [breakout_above, breakdown_below],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def keltner_squeeze_signals(
        self,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1
    ):
        
        """
        Keltner Channels Squeeze Signals
        2 = Squeeze (Channels very narrow - low volatility)
        1 = Expansion (Channels very wide - high volatility)
        0 = Normal
        
        Parameters:
        squeeze_period (int): Period for squeeze detection
        squeeze_threshold (float): Threshold for squeeze identification
        
        """
        
        for upper, lower, middle in zip(self.keltner_upper, self.keltner_lower, self.keltner_middle):
            self._validate_columns(columns = [upper, lower, middle])
            
            channel_width = (self.data[upper] - self.data[lower]) / self.data[middle]
            
            width_percentile = channel_width.rolling(window=squeeze_period).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
            
            squeeze = width_percentile < squeeze_threshold
            expansion = width_percentile > (1 - squeeze_threshold)
            
            self.signals[f'{upper}_{lower}_{middle}_squeeze'] = np.select(
                [squeeze, expansion],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def keltner_trend_signals(
        self
    ):
        
        """
        Keltner Channels Trend Signals
        2 = Strong Uptrend (Price above Middle & Middle rising)
        1 = Strong Downtrend (Price below Middle & Middle falling)
        0 = Weak/No trend
        
        """
        
        for upper, lower, middle in zip(self.keltner_upper, self.keltner_lower, self.keltner_middle):
            self._validate_columns(columns = [upper, lower, middle, self.close_col])
            
            middle_slope = self.data[middle].diff()
            
            strong_uptrend = (
                (self.data[self.close_col] > self.data[middle]) & 
                (middle_slope > 0)
            )
            
            strong_downtrend = (
                (self.data[self.close_col] < self.data[middle]) & 
                (middle_slope < 0)
            )
            
            self.signals[f'{upper}_{lower}_{middle}_trend'] = np.select(
                [strong_uptrend, strong_downtrend],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def keltner_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Keltner Channels Divergence Signals
        2 = Bullish Divergence (Price Lower Low, Channel Position Higher)
        1 = Bearish Divergence (Price Higher High, Channel Position Lower)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for upper, lower in zip(self.keltner_upper, self.keltner_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            channel_position = (self.data[self.close_col] - self.data[lower]) / (self.data[upper] - self.data[lower])
            
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            position_higher = (
                (channel_position > channel_position.shift(lookback)) &
                (channel_position.shift(1) > channel_position.shift(lookback + 1))
            )
            bullish_divergence = price_lower_low & position_higher
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            position_lower = (
                (channel_position < channel_position.shift(lookback)) &
                (channel_position.shift(1) < channel_position.shift(lookback + 1))
            )
            bearish_divergence = price_higher_high & position_lower
            
            self.signals[f'{upper}_{lower}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
       
        return self.signals
    
    def keltner_walk_signals(
        self
    ):
        
        """
        Keltner Channels Walk Signals (Price walking along channels)
        2 = Walking Upper Channel (Strong uptrend)
        1 = Walking Lower Channel (Strong downtrend)
        0 = Not walking channels
        
        """
        
        for upper, lower in zip(self.keltner_upper, self.keltner_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            upper_touch = (self.data[self.close_col] >= self.data[upper] * 0.99)
            lower_touch = (self.data[self.close_col] <= self.data[lower] * 1.01)
            
            walking_upper = (upper_touch & upper_touch.shift(1) & upper_touch.shift(2))
            walking_lower = (lower_touch & lower_touch.shift(1) & lower_touch.shift(2))
            
            self.signals[f'{upper}_{lower}_walk'] = np.select(
                [walking_upper, walking_lower],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def keltner_reversal_signals(
        self
    ):
        
        """
        Keltner Channels Reversal Signals
        2 = Bullish Reversal (Price moves from below Lower to above Middle)
        1 = Bearish Reversal (Price moves from above Upper to below Middle)
        0 = No reversal
        
        """
        
        for upper, lower, middle in zip(self.keltner_upper, self.keltner_lower, self.keltner_middle):
            self._validate_columns(columns = [upper, lower, middle, self.close_col])
            
            bullish_reversal = (
                (self.data[self.close_col] > self.data[middle]) & 
                (self.data[self.close_col].shift(1) <= self.data[lower].shift(1))
            )
            
            bearish_reversal = (
                (self.data[self.close_col] < self.data[middle]) & 
                (self.data[self.close_col].shift(1) >= self.data[upper].shift(1))
            )
            
            self.signals[f'{upper}_{lower}_{middle}_reversal'] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_keltner_signals(
        self,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1,
        lookback: int = 10
    ):
        
        """
        Generate all Keltner Channels signals
        
        """
        
        self.keltner_price_position_signals()
        self.keltner_breakout_signals()
        self.keltner_squeeze_signals(
            squeeze_period = squeeze_period,
            squeeze_threshold = squeeze_threshold
        )
        self.keltner_trend_signals()
        self.keltner_divergence_signals(lookback = lookback)
        self.keltner_walk_signals()
        self.keltner_reversal_signals()
        
        count_removed_rows = self.signals.shape[0] - self.data.shape[0]
        
        if self.prints:
            print('='*50)
            print(self.signals.info())
            print('='*50)   
            print(f'Shape of data {self.signals.shape}')
            print('='*50)
            print(f'{count_removed_rows} rows removed')
            print('='*50)
            
        return self.signals