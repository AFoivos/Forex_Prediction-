import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexStochasticSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
    ):
        
        """
        Class for Stochastic signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("STOCHASTIC SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 stochastic_overbought_oversold_signals \n2 stochastic_crossover_signals \n3 stochastic_divergence_signals \n4 stochastic_momentum_signals \n5 stochastic_reversal_signals \n6 generate_all_stochastic_signals")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.fast_k_param = []
        self.slow_k = []
        self.slow_d = []
        self.slow_k_slope = []
        self.slow_d_slope = []
        
        self.parameters = [14, 3, 3] if parameters is None else parameters
        
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
        Extract Stochastic column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                self.fast_k_param = sublist[0]
                self.slow_k.extend([f'stoch_slowk_{sublist[1]}'])
                self.slow_d.extend([f'stoch_slowd_{sublist[2]}'])
                self.slow_k_slope.extend([f'stoch_slow_{sublist[1]}_slope'])    
                self.slow_d_slope.extend([f'stoch_slowd_{sublist[2]}_slope'])      
        else:
            self.fast_k_param = parameters[0]
            self.slow_k.extend([f'stoch_slowk_{parameters[1]}'])
            self.slow_d.extend([f'stoch_slowd_{parameters[2]}'])
            self.slow_k_slope.extend([f'stoch_slowk_{parameters[1]}_slope'])    
            self.slow_d_slope.extend([f'stoch_slowd_{parameters[2]}_slope']) 
    
    def stochastic_overbought_oversold_signals(
        self, 
        overbought: int = 80, 
        oversold: int = 20
    ):
        """
        Stochastic Overbought/Oversold Signals
        2 = Overbought (Stochastic > 80)
        1 = Oversold (Stochastic < 20)
        0 = Normal (20 <= Stochastic <= 80)
        
        Parameters:
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        for name in self.slow_k:
            self._validate_columns(columns = [name])
        
            overbought_condition = self.data[name] > overbought
            oversold_condition = self.data[name] < oversold
            
            self.signals[f'{name}_overbough'] = np.select(
                [overbought_condition, oversold_condition],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def stochastic_crossover_signals(
        self,
    ):
        
        """
        Stochastic %K/%D Crossover Signals
        2 = Bullish (%K crosses above %D)
        1 = Bearish (%K crosses below %D)
        0 = No crossover
    
        """
        
        for name1, name2 in product(self.slow_k, self.slow_d):
            self._validate_columns(columns = [name1, name2])
        
            bullish_cross = (
                (self.data[name1] > self.data[name2]) &
                (self.data[name1].shift(1) <= self.data[name2].shift(1))
            )
            
            bearish_cross = (
                (self.data[name1] < self.data[name2]) &
                (self.data[name1].shift(1) >= self.data[name2].shift(1))
            )
            
            self.signals[f'{name1}_{name2}_crossover'] = np.select(
                [bullish_cross, bearish_cross],
                [2, 1],
                default = 0
            )
   
        return self.signals
        
    def stochastic_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Stochastic Divergence Signals
        2 = Bullish Divergence (Price Lower Low, Stochastic Higher Low)
        1 = Bearish Divergence (Price Higher High, Stochastic Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for Stochastic
        
        """
        
        for name in self.slow_k:
            self._validate_columns(columns = [name])
           
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            stoch_higher_low = (
                (self.data[name] > self.data[name].shift(lookback)) &
                (self.data[name].shift(1) > self.data[name].shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & stoch_higher_low
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            stoch_lower_high = (
                (self.data[name] < self.data[name].shift(lookback)) &
                (self.data[name].shift(1) < self.data[name].shift(lookback + 1))
            )
            
            bearish_divergence = price_higher_high & stoch_lower_high
            
            self.signals[f'{name}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default=0
            )
       
        return self.signals
    
    def stochastic_momentum_signals(
        self,
    ):
        
        """
        Stochastic Momentum Signals
        2 = Stochastic Rising (Bullish Momentum)
        1 = Stochastic Falling (Bearish Momentum)
        0 = Stochastic Stable
        
        """
        
        for name1 in self.slow_k:
            self._validate_columns(columns = [name1])
         
            k_rising = self.data[name1] > 0
            k_falling = self.data[name1] < 0
            
            self.signals[f'{name1}_momentum'] = np.select(
                [k_rising, k_falling],
                [2, 1],
                default = 0
            )
            
        for name2 in self.slow_d:
            self._validate_columns(columns = [name2])
         
            d_rising = self.data[name2] > 0
            d_falling = self.data[name2] < 0
            
            self.signals[f'{name2}_momentum'] = np.select(
                [d_rising, d_falling],
                [2, 1],
                default = 0
            )
     
        return self.signals
    
    def stochastic_reversal_signals(
        self,
        overbought: int = 80,
        oversold: int = 20
    ):
        
        """
        Stochastic Trend Reversal Signals
        2 = Bullish Reversal (Stochastic exits oversold <20 to >20)
        1 = Bearish Reversal (Stochastic exits overbought >80 to <80)
        0 = No reversal
        
        Parameters:
        overbought (int): Overbought threshold
        oversold (int): Oversold threshold
        
        """
        
        for name in self.slow_k:
            self._validate_columns(columns = [name])
          
            bullish_reversal = (
                (self.data[name] > oversold) &
                (self.data[name].shift(1) <= oversold )
            )
            
            bearish_reversal = (
                (self.data[name] < overbought) &
                (self.data[name] <= overbought)
            )
            
            self.signals[f'{name}_reversal'] = np.select(
                [bullish_reversal, bearish_reversal],
                [2, 1],
                default = 0 
            )
            
        return self.signals
        
    def generate_all_stochastic_signals(
        self,
        overbought: int = 80,
        oversold: int = 20,
        lookback: int = 10
    ):
        
        """
        Generate all Stochastic signals
        
        """
        
        self.stochastic_overbought_oversold_signals(
            overbought = overbought,
            oversold = oversold
        )
        self.stochastic_crossover_signals()
        self.stochastic_divergence_signals(lookback = lookback)
        self.stochastic_momentum_signals()
        self.stochastic_reversal_signals(
            overbought = overbought,
            oversold = oversold
        )
        
        count_removed_rows = self.signals.shape[0] - self.data.shape[0]
        
        print('='*50)
        print('Data Info')
        print(self.signals.info())
        print('='*50)   
        print(f'Shape of data {self.signals.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.signals