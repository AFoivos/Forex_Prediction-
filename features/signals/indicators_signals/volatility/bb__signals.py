import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexBollingerBandsSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
    ):
        
        """
        Class for Bollinger Bands signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("BOLLINGER BANDS SIGNAL GENERATION")
            print("="*50)
            print("Available functions: \n1 bb_price_position_signals \n2 bb_band_width_signals \n3 bb_squeeze_signals \n4 bb_breakout_signals \n5 bb_momentum_signals \n6 bb_divergence_signals \n7 generate_all_bb_signals")
            print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = {}
        signals_names= [
            'trend_direction',          # Κατεύθυνση τάσης  
            'momentum',                 # Ορμή
            'volatility',               # Μεταβλητότητα
            'breakout',                 # Εκρήξεις
            'divergence',               # Αποκλίσεις
            'price_position',           # Θέση τιμής
            'squeeze',                  # Συμπίεση
        ]

        for name in signals_names:
            self.signals[name] = pd.DataFrame(
                {self.close_col: self.data[self.close_col]},
                index = self.data.index
            )
        
        self.bb_upper = []
        self.bb_middle = []
        self.bb_lower = []
        
        self.parameters = [20, 2.0, 2.0] if parameters is None else parameters
        
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
        Extract Bollinger Bands column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                timeperiod = sublist[0]
                nbdevup = sublist[1]
                nbdevdn = sublist[2]
                
                col_name = f'bb_{timeperiod}_{nbdevup}_{nbdevdn}'
                
                self.bb_upper.extend([f'{col_name}_upper'])
                self.bb_middle.extend([f'{col_name}_middle'])
                self.bb_lower.extend([f'{col_name}_lower'])
        else:
            timeperiod = parameters[0]
            nbdevup = parameters[1]
            nbdevdn = parameters[2]
            
            col_name = f'bb_{timeperiod}_{nbdevup}_{nbdevdn}'
            
            self.bb_upper.extend([f'{col_name}_upper'])
            self.bb_middle.extend([f'{col_name}_middle'])
            self.bb_lower.extend([f'{col_name}_lower'])
    
    def bb_price_position_signals(self):
        """
        Bollinger Bands Price Position Signals
        2 = Price above Upper Band (Overbought)
        1 = Price below Lower Band (Oversold)
        0 = Price between Bands (Normal)
        
        """
        
        for upper, lower in zip(self.bb_upper, self.bb_lower):
            self._validate_columns(columns = [upper, lower])
            
            above_upper = self.data[self.close_col] > self.data[upper]
            below_lower = self.data[self.close_col] < self.data[lower]
            
            col_name = upper.replace('_upper', '')
            self.signals['price_position'][f'{col_name}_above_below'] = np.select(
                [above_upper, below_lower],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def bb_band_width_signals(
        self,
        wide_threshold: float = 0.05,
        narrow_threshold: float = 0.02
    ):
        """
        Bollinger Bands Width Signals
        2 = Wide Bands (High Volatility)
        1 = Narrow Bands (Low Volatility)
        0 = Normal Band Width
        
        Parameters:
        wide_threshold (float): Wide bands threshold
        narrow_threshold (float): Narrow bands threshold
        
        """
        
        for upper, lower, middle in zip(self.bb_upper, self.bb_lower, self.bb_middle):
            self._validate_columns(columns = [upper, lower, middle])
            
            band_width = (self.data[upper] - self.data[lower]) / self.data[middle]
            
            wide_bands = band_width > wide_threshold
            narrow_bands = band_width < narrow_threshold
            
            col_name = upper.replace('_upper', '')
            self.signals['volatility'][f'{col_name}_wide_narrow'] = np.select(
                [wide_bands, narrow_bands],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def bb_squeeze_signals(
        self,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1
    ):
        """
        Bollinger Bands Squeeze Signals
        2 = Squeeze (Bands very narrow - low volatility)
        1 = Expansion (Bands very wide - high volatility)
        0 = Normal
        
        Parameters:
        squeeze_period (int): Period for squeeze detection
        squeeze_threshold (float): Threshold for squeeze identification
        
        """
        
        for upper, lower, middle in zip(self.bb_upper, self.bb_lower, self.bb_middle):
            self._validate_columns(columns = [upper, lower, middle])
            
            band_width = (self.data[upper] - self.data[lower]) / self.data[middle]
            
            width_percentile = band_width.rolling(window=squeeze_period).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
            
            squeeze = width_percentile < squeeze_threshold
            expansion = width_percentile > (1 - squeeze_threshold)
            
            col_name = upper.replace('_upper', '')
            self.signals['squeeze'][f'{col_name}_squeeze_expansion'] = np.select(
                [squeeze, expansion],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def bb_breakout_signals(
        self
    ):
        
        """
        Bollinger Bands Breakout Signals
        2 = Breakout above Upper Band
        1 = Breakdown below Lower Band
        0 = No breakout
        
        """
        
        for upper, lower in zip(self.bb_upper, self.bb_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            breakout_above = (
                (self.data[self.close_col] > self.data[upper]) & 
                (self.data[self.close_col].shift(1) <= self.data[upper].shift(1))
            )
            
            breakdown_below = (
                (self.data[self.close_col] < self.data[lower]) & 
                (self.data[self.close_col].shift(1) >= self.data[lower].shift(1))
            )
            
            col_name = upper.replace('_upper', '')
            self.signals['breakout'][f'{col_name}_breakout'] = np.select(
                [breakout_above, breakdown_below],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def bb_momentum_signals(
        self
    ): 
        
        """
        Bollinger Bands Momentum Signals
        2 = Price in Upper Half (Strong uptrend)
        1 = Price in Lower Half (Strong downtrend)
        0 = Price around Middle Band (Neutral)
        
        """
        
        for upper, lower, middle in zip(self.bb_upper, self.bb_lower, self.bb_middle):
            self._validate_columns(columns = [upper, lower, middle, self.close_col])
            
            band_position = (self.data[self.close_col] - self.data[lower]) / (self.data[upper] - self.data[lower])
            
            upper_half = band_position > 0.7  
            lower_half = band_position < 0.3  
            
            col_name = upper.replace('_upper', '')
            self.signals['momentum'][f'{col_name}_upper_lower_half'] = np.select(
                [upper_half, lower_half],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def bb_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Bollinger Bands %B Divergence Signals
        2 = Bullish Divergence (Price Lower Low, %B Higher Low)
        1 = Bearish Divergence (Price Higher High, %B Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for upper, lower in zip(self.bb_upper, self.bb_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            percent_b = (self.data[self.close_col] - self.data[lower]) / (self.data[upper] - self.data[lower])
            
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            percent_b_higher_low = (
                (percent_b > percent_b.shift(lookback)) &
                (percent_b.shift(1) > percent_b.shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & percent_b_higher_low
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            percent_b_lower_high = (
                (percent_b < percent_b.shift(lookback)) &
                (percent_b.shift(1) < percent_b.shift(lookback + 1))
            )
            bearish_divergence = price_higher_high & percent_b_lower_high
            
            col_name = upper.replace('_upper', '')
            self.signals['divergence'][f'{col_name}_bullish_bearish'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
       
        return self.signals
    
    def bb_walk_signals(
        self
    ):
        
        """
        Bollinger Bands Walk Signals (Price walking along bands)
        2 = Walking Upper Band (Strong uptrend)
        1 = Walking Lower Band (Strong downtrend)
        0 = Not walking bands
        
        """
        
        for upper, lower in zip(self.bb_upper, self.bb_lower):
            self._validate_columns(columns = [upper, lower, self.close_col])
            
            upper_touch = (self.data[self.close_col] >= self.data[upper] * 0.99)
            lower_touch = (self.data[self.close_col] <= self.data[lower] * 1.01)
            
            walking_upper = (upper_touch & upper_touch.shift(1) & upper_touch.shift(2))
            walking_lower = (lower_touch & lower_touch.shift(1) & lower_touch.shift(2))
            
            col_name = upper.replace('_upper', '')
            self.signals['trend_direction'][f'{col_name}_upper_lower_walking'] = np.select(
                [walking_upper, walking_lower],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_bb_signals(
        self,
        wide_threshold: float = 0.05,
        narrow_threshold: float = 0.02,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1,
        lookback: int = 10
    ):
        
        """
        Generate all Bollinger Bands signals
        
        """
        
        self.bb_price_position_signals()
        self.bb_band_width_signals(
            wide_threshold=wide_threshold,
            narrow_threshold=narrow_threshold
        )
        self.bb_squeeze_signals(
            squeeze_period = squeeze_period,
            squeeze_threshold = squeeze_threshold
        )
        self.bb_breakout_signals()
        self.bb_momentum_signals()
        self.bb_divergence_signals(lookback = lookback)
        self.bb_walk_signals()
        
        count_removed_rows = 0
        for name in self.signals.keys():
            count_removed_rows += self.signals[name].shape[0] - self.data.shape[0]
        
        if self.prints:
            print('='*50)
            for name in self.signals.keys():
                print(self.signals[name].info())
            print('='*50)   
            for name in self.signals.keys():
                print(f'Shape of {name} data {self.signals[name].shape}')
            print('='*50)   
            print(f'{count_removed_rows} rows removed')
            print('='*50)
        
        return self.signals