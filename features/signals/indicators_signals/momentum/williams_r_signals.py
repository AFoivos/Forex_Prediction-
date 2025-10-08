import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexWilliamsRSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
    ):
        
        """
        Class for Williams %R signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("WILLIAMS %R SIGNAL GENERATION")
            print("="*50)
            print("Available functions: \n1 williams_overbought_oversold_signals \n2 williams_momentum_signals \n3 williams_reversal_signals \n4 williams_divergence_signals \n5 generate_all_williams_signals")
            print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.williams_r = []
        self.williams_r_slope = []
        
        self.parameters = [14, 21, 28] if parameters is None else parameters
        
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
        Extract Williams %R column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    self.williams_r.extend([f'williams_r_{period}'])
                    self.williams_r_slope.extend([f'williams_r_{period}_slope'])
        else:
            for period in parameters:
                self.williams_r.extend([f'williams_r_{period}'])
                self.williams_r_slope.extend([f'williams_r_{period}_slope'])
    
    def williams_overbought_oversold_signals(
        self, 
        overbought: int = -20, 
        oversold: int = -80
    ):
        
        """
        Williams %R Overbought/Oversold Signals
        2 = Overbought (Williams %R > -20)
        1 = Oversold (Williams %R < -80)
        0 = Normal (-80 <= Williams %R <= -20)
        
        Parameters:
        overbought (int): Overbought threshold (default: -20)
        oversold (int): Oversold threshold (default: -80)
        
        """
        
        for name in self.williams_r:
            self._validate_columns(columns = [name])
        
            overbought_condition = self.data[name] > overbought  
            oversold_condition = self.data[name] < oversold      
            
            self.signals[f'{name}_overbought_oversold'] = np.select(
                [overbought_condition, oversold_condition],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def williams_momentum_signals(
        self
    ):
        
        """
        Williams %R Momentum Signals
        2 = Williams %R Rising (Bullish Momentum - moving toward 0)
        1 = Williams %R Falling (Bearish Momentum - moving toward -100)
        0 = Williams %R Stable
        
        """
        
        for name in self.williams_r_slope:
            self._validate_columns(columns = [name])
         
            williams_rising = self.data[name] > 0
            williams_falling = self.data[name] < 0
            
            self.signals[f'{name}_momentum'] = np.select(
                [williams_rising, williams_falling],
                [2, 1],
                default = 0
            )
     
        return self.signals
    
    def williams_reversal_signals(
        self,
        overbought: int = -20,
        oversold: int = -80
    ):
        
        """
        Williams %R Trend Reversal Signals
        2 = Bullish Reversal (Williams %R exits oversold <-80 to >-80)
        1 = Bearish Reversal (Williams %R exits overbought >-20 to <-20)
        0 = No reversal
        
        Parameters:
        overbought (int): Overbought threshold (default: -20)
        oversold (int): Oversold threshold (default: -80)
        
        """
        
        for name in self.williams_r:
            self._validate_columns(columns = [name])
          
            bullish_reversal = (
                (self.data[name] > oversold) &
                (self.data[name].shift(1) <= oversold)
            )
            
            bearish_reversal = (
                (self.data[name] < overbought) &
                (self.data[name].shift(1) >= overbought)
            )
            
            self.signals[f'{name}_reversal'] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default = 0
            )
            
        return self.signals
    
    def williams_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Williams %R Divergence Signals
        2 = Bullish Divergence (Price Lower Low, Williams %R Higher Low)
        1 = Bearish Divergence (Price Higher High, Williams %R Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for name in self.williams_r:
            self._validate_columns(columns = [name])
           
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            williams_higher_low = (
                (self.data[name] > self.data[name].shift(lookback)) & 
                (self.data[name].shift(1) > self.data[name].shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & williams_higher_low
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            williams_lower_high = (
                (self.data[name] < self.data[name].shift(lookback)) &  
                (self.data[name].shift(1) < self.data[name].shift(lookback + 1))
            )
            
            bearish_divergence = price_higher_high & williams_lower_high
            
            self.signals[f'{name}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
       
        return self.signals
    
    def williams_extreme_signals(
        self,
        extreme_overbought: int = -10,
        extreme_oversold: int = -90
    ):
        
        """
        Williams %R Extreme Zone Signals
        2 = Extreme Overbought (Williams %R > -10)
        1 = Extreme Oversold (Williams %R < -90)  
        0 = Not in extreme zone
        
        Parameters:
        extreme_overbought (int): Extreme overbought threshold (default: -10)
        extreme_oversold (int): Extreme oversold threshold (default: -90)
        
        """
        
        for name in self.williams_r:
            self._validate_columns(columns = [name])
        
            extreme_overbought_condition = self.data[name] > extreme_overbought
            extreme_oversold_condition = self.data[name] < extreme_oversold
            
            self.signals[f'{name}_extreme'] = np.select(
                [extreme_overbought_condition, extreme_oversold_condition],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_williams_signals(
        self,
        overbought: int = -20,
        oversold: int = -80,
        extreme_overbought: int = -10,
        extreme_oversold: int = -90,
        lookback: int = 10
    ):
        
        """
        Generate all Williams %R signals
        
        """
        
        self.williams_overbought_oversold_signals(
            overbought = overbought,
            oversold = oversold
        )
        self.williams_momentum_signals()
        self.williams_reversal_signals(
            overbought = overbought,
            oversold = oversold
        )
        self.williams_divergence_signals(lookback=lookback)
        self.williams_extreme_signals(
            extreme_overbought = extreme_overbought,
            extreme_oversold = extreme_oversold
        )
        
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