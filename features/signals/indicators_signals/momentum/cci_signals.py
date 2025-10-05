import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union
from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexCCISignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
    ):
        
        """
        Class for CCI signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("CCI SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 cci_overbought_oversold_signals \n2 cci_momentum_signals \n3 cci_reversal_signals \n4 cci_divergence_signals \n5 cci_zero_line_signals \n6 generate_all_cci_signals")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index = self.data.index
        )
        
        self.cci = []
        self.cci_slope = []
        
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
        Extract CCI column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    self.cci.extend([f'cci_{period}'])
                    self.cci_slope.extend([f'cci_{period}_slope'])
        else:
            for period in parameters:
                self.cci.extend([f'cci_{period}'])
                self.cci_slope.extend([f'cci_{period}_slope'])
    
    def cci_overbought_oversold_signals(
        self, 
        overbought: int = 100, 
        oversold: int = -100
    ):
        
        """
        CCI Overbought/Oversold Signals
        2 = Overbought (CCI > +100)
        1 = Oversold (CCI < -100)
        0 = Normal (-100 <= CCI <= +100)
        
        Parameters:
        overbought (int): Overbought threshold (default: +100)
        oversold (int): Oversold threshold (default: -100)
        
        """
        
        for name in self.cci:
            self._validate_columns(columns=[name])
        
            overbought_condition = self.data[name] > overbought
            oversold_condition = self.data[name] < oversold
            
            self.signals[f'{name}_overbought_oversold'] = np.select(
                [overbought_condition, oversold_condition],
                [2, 1],
                default=0
            )
         
        return self.signals
    
    def cci_momentum_signals(
        self
    ):
        
        """
        CCI Momentum Signals
        2 = CCI Rising (Bullish Momentum)
        1 = CCI Falling (Bearish Momentum)
        0 = CCI Stable
        
        """
        
        for name in self.cci_slope:
            self._validate_columns(columns=[name])
         
            # Rising: slope > 0 (bullish momentum)
            cci_rising = self.data[name] > 0
            # Falling: slope < 0 (bearish momentum)
            cci_falling = self.data[name] < 0
            
            self.signals[f'{name}_momentum'] = np.select(
                [cci_rising, cci_falling],
                [2, 1],
                default=0
            )
     
        return self.signals
    
    def cci_reversal_signals(
        self,
        overbought: int = 100,
        oversold: int = -100
    ):
        
        """
        CCI Trend Reversal Signals
        2 = Bullish Reversal (CCI exits oversold <-100 to >-100)
        1 = Bearish Reversal (CCI exits overbought >+100 to <+100)
        0 = No reversal
        
        Parameters:
        overbought (int): Overbought threshold (default: +100)
        oversold (int): Oversold threshold (default: -100)
        
        """
        
        for name in self.cci:
            self._validate_columns(columns=[name])
          
            # Bullish: exits oversold zone
            bullish_reversal = (
                (self.data[name] > oversold) &
                (self.data[name].shift(1) <= oversold)
            )
            
            # Bearish: exits overbought zone  
            bearish_reversal = (
                (self.data[name] < overbought) &
                (self.data[name].shift(1) >= overbought)
            )
            
            self.signals[f'{name}_reversal'] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default=0
            )
            
        return self.signals
    
    def cci_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        CCI Divergence Signals
        2 = Bullish Divergence (Price Lower Low, CCI Higher Low)
        1 = Bearish Divergence (Price Higher High, CCI Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for name in self.cci:
            self._validate_columns(columns=[name])
           
            # Bullish Divergence: Price Lower Low, CCI Higher Low
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            cci_higher_low = (
                (self.data[name] > self.data[name].shift(lookback)) &
                (self.data[name].shift(1) > self.data[name].shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & cci_higher_low
            
            # Bearish Divergence: Price Higher High, CCI Lower High
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            cci_lower_high = (
                (self.data[name] < self.data[name].shift(lookback)) &
                (self.data[name].shift(1) < self.data[name].shift(lookback + 1))
            )
            
            bearish_divergence = price_higher_high & cci_lower_high
            
            self.signals[f'{name}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default=0
            )
       
        return self.signals
    
    def cci_zero_line_signals(
        self
    ):
        
        """
        CCI Zero Line Crossover Signals
        2 = CCI crosses above Zero Line (bullish)
        1 = CCI crosses below Zero Line (bearish)
        0 = No crossover
        
        """
        
        for name in self.cci:
            self._validate_columns(columns=[name])

            above_zero = (
                (self.data[name] > 0) & 
                (self.data[name].shift(1) <= 0)
            )

            below_zero = (
                (self.data[name] < 0) & 
                (self.data[name].shift(1) >= 0)
            )

            self.signals[f'{name}_zero_cross'] = np.select(
                [above_zero, below_zero],
                [2, 1],
                default=0
            )
                    
        return self.signals
    
    def cci_extreme_signals(
        self,
        extreme_overbought: int = 200,
        extreme_oversold: int = -200
    ):
        """
        CCI Extreme Zone Signals
        2 = Extreme Overbought (CCI > +200)
        1 = Extreme Oversold (CCI < -200)  
        0 = Not in extreme zone
        
        Parameters:
        extreme_overbought (int): Extreme overbought threshold (default: +200)
        extreme_oversold (int): Extreme oversold threshold (default: -200)
        
        """
        
        for name in self.cci:
            self._validate_columns(columns=[name])
        
            extreme_overbought_condition = self.data[name] > extreme_overbought
            extreme_oversold_condition = self.data[name] < extreme_oversold
            
            self.signals[f'{name}_extreme'] = np.select(
                [extreme_overbought_condition, extreme_oversold_condition],
                [2, 1],
                default=0
            )
         
        return self.signals
        
    def generate_all_cci_signals(
        self,
        overbought: int = 100,
        oversold: int = -100,
        extreme_overbought: int = 200,
        extreme_oversold: int = -200,
        lookback: int = 10
    ):
        
        """
        Generate all CCI signals
        
        """
        
        self.cci_overbought_oversold_signals(
            overbought=overbought,
            oversold=oversold
        )
        self.cci_momentum_signals()
        self.cci_reversal_signals(
            overbought=overbought,
            oversold=oversold
        )
        self.cci_divergence_signals(lookback=lookback)
        self.cci_zero_line_signals()
        self.cci_extreme_signals(
            extreme_overbought=extreme_overbought,
            extreme_oversold=extreme_oversold
        )
        
        count_removed_rows = self.signals.shape[0] - self.data.shape[0]
        
        print('='*50)
        print(self.signals.info())
        print('='*50)   
        print(f'Shape of data {self.signals.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.signals