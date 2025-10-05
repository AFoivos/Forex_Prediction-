# atr_signals.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexATRSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
    ):
        
        """
        Class for ATR signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
        print("="*50)
        print("ATR SIGNAL GENERATION")
        print("="*50)
        print("Available functions: \n1 atr_volatility_signals \n2 atr_breakout_signals \n3 atr_trend_strength_signals \n4 atr_squeeze_signals \n5 atr_expansion_contraction_signals \n6 generate_all_atr_signals")
        print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.atr = []
        self.atr_slope = []
        
        self.parameters = [10, 14, 21, 28] if parameters is None else parameters
        
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
        Extract ATR column names based on parameters
        
        """
        is_nested = self._is_nested_list(parameters)
        
        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    self.atr.extend([f'atr_{period}'])
                    self.atr_slope.extend([f'atr_{period}_slope'])
        else:
            for period in parameters:
                self.atr.extend([f'atr_{period}'])
                self.atr_slope.extend([f'atr_{period}_slope'])
    
    def atr_volatility_signals(
        self, 
        high_volatility_multiplier: float = 1.5,
        low_volatility_multiplier: float = 0.5
    ):
        
        """
        ATR Volatility Level Signals
        2 = High Volatility (ATR > 2.0 * SMA of ATR)
        1 = Low Volatility (ATR < 0.5 * SMA of ATR)
        0 = Normal Volatility
        
        Parameters:
        high_volatility_multiplier (float): High volatility threshold multiplier
        low_volatility_multiplier (float): Low volatility threshold multiplier
        
        """
        
        for name in self.atr:
            self._validate_columns(columns = [name])
            
            atr_ma = self.data[name].rolling(window = 10).mean()

            high_volatility = self.data[name] > (atr_ma * high_volatility_multiplier)
            low_volatility = self.data[name] < (atr_ma * low_volatility_multiplier)
            
            self.signals[f'{name}_volatility_level'] = np.select(
                [high_volatility, low_volatility],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def atr_breakout_signals(
        self,
        lookback_period: int = 20
    ):
        
        """
        ATR Breakout Signals
        2 = Volatility Breakout (ATR > max of recent ATR)
        1 = Volatility Collapse (ATR < min of recent ATR)
        0 = Normal volatility
        
        Parameters:
        lookback_period (int): Lookback period for breakout detection
        
        """
        
        for name in self.atr:
            self._validate_columns(columns = [name])
            
            atr_high = self.data[name].rolling(window=lookback_period).max()
            atr_low = self.data[name].rolling(window=lookback_period).min()
            
            volatility_breakout = self.data[name] > atr_high.shift(1)
            volatility_collapse = self.data[name] < atr_low.shift(1)
            
            self.signals[f'{name}_breakout'] = np.select(
                [volatility_breakout, volatility_collapse],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def atr_trend_strength_signals(
        self,
        strong_trend_threshold: float = 1.5,
        weak_trend_threshold: float = 0.7
    ):
        
        """
        ATR Trend Strength Signals (based on ATR expansion/contraction)
        2 = Strong Trend (ATR expanding rapidly)
        1 = Weak Trend/Consolidation (ATR contracting)
        0 = Moderate Trend
        
        Parameters:
        strong_trend_threshold (float): Strong trend threshold for ATR slope
        weak_trend_threshold (float): Weak trend threshold for ATR slope
        
        """
        
        for name in self.atr_slope:
            self._validate_columns(columns = [name])
            
            atr_value = self.data[name.replace('_slope', '')]
            normalized_slope = self.data[name] / atr_value
            
            strong_trend = normalized_slope > strong_trend_threshold
            weak_trend = normalized_slope < weak_trend_threshold
            
            self.signals[f'{name}_trend_strength'] = np.select(
                [strong_trend, weak_trend],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def atr_squeeze_signals(
        self,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1
    ):
        
        """
        ATR Squeeze Signals (Low volatility periods before big moves)
        2 = Squeeze (ATR at multi-period low)
        1 = Expansion (ATR at multi-period high)
        0 = Normal
        
        Parameters:
        squeeze_period (int): Period for squeeze detection
        squeeze_threshold (float): Threshold for squeeze identification
        
        """
        
        for name in self.atr:
            self._validate_columns(columns = [name])
            
            atr_percentile = self.data[name].rolling(window=squeeze_period).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) if x.max() != x.min() else 0.5
            )
            
            squeeze = atr_percentile < squeeze_threshold  
            expansion = atr_percentile > (1 - squeeze_threshold)  
            
            self.signals[f'{name}_squeeze'] = np.select(
                [squeeze, expansion],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def atr_expansion_contraction_signals(
        self
    ):
        
        """
        ATR Expansion/Contraction Signals
        2 = ATR Expanding (Volatility increasing)
        1 = ATR Contracting (Volatility decreasing)
        0 = ATR Stable
        
        """
        
        for name in self.atr_slope:
            self._validate_columns(columns=[name])
            
            atr_expanding = self.data[name] > 0
            atr_contracting = self.data[name] < 0
            
            self.signals[f'{name}_expansion'] = np.select(
                [atr_expanding, atr_contracting],
                [2, 1],
                default = 0
            )
         
        return self.signals
    
    def atr_support_resistance_signals(
        self,
        atr_multiplier: float = 2.0
    ):
        
        """
        ATR-based Support/Resistance Break Signals
        2 = Price breaks above ATR-based resistance
        1 = Price breaks below ATR-based support
        0 = Price within ATR range
        
        Parameters:
        atr_multiplier (float): Multiplier for ATR-based levels
        
        """
        
        for name in self.atr:
            self._validate_columns(columns = [name])
            
            atr_value = self.data[name]
            prev_close = self.data[self.close_col].shift(1)
            
            resistance = prev_close + (atr_value * atr_multiplier)
            support = prev_close - (atr_value * atr_multiplier)
            
            break_above_resistance = self.data[self.close_col] > resistance
            break_below_support = self.data[self.close_col] < support
            
            self.signals[f'{name}_sr_break'] = np.select(
                [break_above_resistance, break_below_support],
                [2, 1],
                default = 0
            )
         
        return self.signals
        
    def generate_all_atr_signals(
        self,
        high_volatility_multiplier: float = 1.5,
        low_volatility_multiplier: float = 0.5,
        lookback_period: int = 20,
        strong_trend_threshold: float = 1.5,
        weak_trend_threshold: float = 0.7,
        squeeze_period: int = 20,
        squeeze_threshold: float = 0.1,
        atr_multiplier: float = 2.0
    ):
        
        """
        Generate all ATR signals
        
        """
        
        self.atr_volatility_signals(
            high_volatility_multiplier = high_volatility_multiplier,
            low_volatility_multiplier = low_volatility_multiplier
        )
        self.atr_breakout_signals(lookback_period = lookback_period)
        self.atr_trend_strength_signals(
            strong_trend_threshold = strong_trend_threshold,
            weak_trend_threshold = weak_trend_threshold
        )
        self.atr_squeeze_signals(
            squeeze_period = squeeze_period,
            squeeze_threshold = squeeze_threshold
        )
        self.atr_expansion_contraction_signals()
        self.atr_support_resistance_signals(atr_multiplier = atr_multiplier)
        
        count_removed_rows = self.signals.shape[0] - self.data.shape[0]
        
        print('='*50)
        print(self.signals.info())
        print('='*50)   
        print(f'Shape of data {self.signals.shape}')
        print('='*50)
        print(f'{count_removed_rows} rows removed')
        print('='*50)
        
        return self.signals