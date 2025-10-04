# momentum_signals.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexMomentumSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
    ):
        
        """
        Class for Momentum signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("MOMENTUM SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 momentum_direction_signals \n2 momentum_momentum_signals \n3 momentum_zero_line_signals \n4 momentum_divergence_signals \n5 momentum_acceleration_signals \n6 generate_all_momentum_signals")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.momentum = []
        self.momentum_slope = []
        
        self.parameters = [10, 14, 20] if parameters is None else parameters
        
        self._validate_columns()
        self._extract_column_names(parameters=parameters)
        
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
        Extract Momentum column names based on parameters
        
        """
        
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    self.momentum.extend([f'momentum_{period}'])
                    self.momentum_slope.extend([f'momentum_{period}_slope'])
        else:
            for period in parameters:
                self.momentum.extend([f'momentum_{period}'])
                self.momentum_slope.extend([f'momentum_{period}_slope'])
    
    def momentum_direction_signals(
        self
    ):
        
        """
        Momentum Direction Signals
        2 = Positive Momentum (Momentum > 0)
        1 = Negative Momentum (Momentum < 0)
        0 = Neutral (Momentum = 0)
        
        """
        
        for name in self.momentum:
            self._validate_columns(columns = [name])
        
            positive_momentum = self.data[name] > 0
            negative_momentum = self.data[name] < 0
            
            self.signals[f'{name}_direction'] = np.select(
                [positive_momentum, negative_momentum],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def momentum_momentum_signals(self):
        """
        Momentum of Momentum Signals (Acceleration/Deceleration)
        2 = Momentum Accelerating (Slope > 0)
        1 = Momentum Decelerating (Slope < 0)
        0 = Momentum Stable (Slope = 0)
        
        """
        
        for name in self.momentum_slope:
            self._validate_columns(columns = [name])
         
            # Accelerating: slope > 0
            momentum_accelerating = self.data[name] > 0
            # Decelerating: slope < 0
            momentum_decelerating = self.data[name] < 0
            
            self.signals[f'{name}_acceleration'] = np.select(
                [momentum_accelerating, momentum_decelerating],
                [2, 1],
                default = 0
            )
     
        return self.signals
    
    def momentum_zero_line_signals(
        self
    ):
        
        """
        Momentum Zero Line Crossover Signals
        2 = Momentum crosses above Zero Line (bullish)
        1 = Momentum crosses below Zero Line (bearish)
        0 = No crossover
        
        """
        
        for name in self.momentum:
            self._validate_columns(columns = [name])

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
                default = 0
            )
                    
        return self.signals
    
    def momentum_divergence_signals(
        self,
        lookback: int = 10
    ):
        
        """
        Momentum Divergence Signals
        2 = Bullish Divergence (Price Lower Low, Momentum Higher Low)
        1 = Bearish Divergence (Price Higher High, Momentum Lower High)
        0 = No Divergence
        
        Parameters:
        lookback (int): Lookback period for divergence detection
        
        """
        
        for name in self.momentum:
            self._validate_columns(columns = [name])
           
            price_lower_low = (
                (self.data[self.close_col] < self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) < self.data[self.close_col].shift(lookback + 1))
            )
            
            momentum_higher_low = (
                (self.data[name] > self.data[name].shift(lookback)) &
                (self.data[name].shift(1) > self.data[name].shift(lookback + 1))
            )
            
            bullish_divergence = price_lower_low & momentum_higher_low
            
            price_higher_high = (
                (self.data[self.close_col] > self.data[self.close_col].shift(lookback)) &
                (self.data[self.close_col].shift(1) > self.data[self.close_col].shift(lookback + 1))
            )
            
            momentum_lower_high = (
                (self.data[name] < self.data[name].shift(lookback)) &
                (self.data[name].shift(1) < self.data[name].shift(lookback + 1))
            )
            
            bearish_divergence = price_higher_high & momentum_lower_high
            
            self.signals[f'{name}_divergence'] = np.select(
                [bullish_divergence, bearish_divergence],
                [2, 1],
                default = 0
            )
       
        return self.signals
    
    def momentum_extreme_signals(
        self,
        extreme_bullish: float = 5.0,
        extreme_bearish: float = -5.0
    ):
        
        """
        Momentum Extreme Zone Signals
        2 = Extreme Bullish (Momentum > +5.0)
        1 = Extreme Bearish (Momentum < -5.0)  
        0 = Normal momentum
        
        Parameters:
        extreme_bullish (float): Extreme bullish threshold (default: +5.0)
        extreme_bearish (float): Extreme bearish threshold (default: -5.0)
        
        """
        
        for name in self.momentum:
            self._validate_columns(columns = [name])
        
            extreme_bullish_condition = self.data[name] > extreme_bullish
            extreme_bearish_condition = self.data[name] < extreme_bearish
            
            self.signals[f'{name}_extreme'] = np.select(
                [extreme_bullish_condition, extreme_bearish_condition],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def momentum_trend_strength_signals(
        self,
        strong_threshold: float = 3.0,
        weak_threshold: float = 1.0
    ):
        
        """
        Momentum Trend Strength Signals
        2 = Strong Trend (|Momentum| > 3.0)
        1 = Weak Trend (|Momentum| < 1.0)
        0 = Moderate Trend (1.0 <= |Momentum| <= 3.0)
        
        Parameters:
        strong_threshold (float): Strong trend threshold (default: 3.0)
        weak_threshold (float): Weak trend threshold (default: 1.0)
        
        """
        
        for name in self.momentum:
            self._validate_columns(columns = [name])
        
            strong_trend = abs(self.data[name]) > strong_threshold
            weak_trend = abs(self.data[name]) < weak_threshold
            
            self.signals[f'{name}_trend_strength'] = np.select(
                [strong_trend, weak_trend],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_momentum_signals(
        self,
        extreme_bullish: float = 5.0,
        extreme_bearish: float = -5.0,
        strong_threshold: float = 3.0,
        weak_threshold: float = 1.0,
        lookback: int = 10
    ):
        
        """
        Generate all Momentum signals
        
        """
        
        self.momentum_direction_signals()
        self.momentum_momentum_signals()
        self.momentum_zero_line_signals()
        self.momentum_divergence_signals(lookback = lookback)
        self.momentum_extreme_signals(
            extreme_bullish=extreme_bullish,
            extreme_bearish=extreme_bearish
        )
        self.momentum_trend_strength_signals(
            strong_threshold=strong_threshold,
            weak_threshold=weak_threshold
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