import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexADXSignals:
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
    ):
        """
        Class for ADX signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("ADX SIGNAL GENERATION")
            print("="*50)
            print(" Available Fuctions: \n1 adx_trend_strength_signals \n2 adx_direction_signals \n3 adx_di_crossover_signals \n4 adx_slope_signals \n5 adx_comprehensive_signals \n6 generate_all_adx_signals")
            print("="*50)
            
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.adx_names  = []
        self.adx_plus = []
        self.adx_minus = []
        self.adx_slopes = []
        self.adx_plus_slopes = []
        self.adx_minus_slopes = []
        
        self.adx_parameters = [14, 21, 28] if parameters is None else parameters
        
        self._validate_columns()
        self._extract_column_names(parameters = self.adx_parameters)
    
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
        Extract MA column names based on parameters
        
        """
        
        is_nested = self._is_nested_list(parameters)
        
        def _validate():
            self._validate_columns(
                        [self.adx_names[-1], 
                        self.adx_plus[-1],
                        self.adx_minus[-1], 
                        self.adx_slopes[-1],
                        self.adx_plus_slopes[-1], 
                        self.adx_minus_slopes[-1]]
                    )

        if is_nested:
            for sublist in parameters:
                for period in sublist:
                    self.adx_names.extend([f'adx_{period}'])
                    self.adx_plus.extend([f'adx_{period}_plus'])
                    self.adx_minus.extend([f'adx_{period}_minus'])
                    self.adx_slopes.extend([f'adx_{period}_slope'])
                    self.adx_plus_slopes.extend([f'adx_{period}_plus_di_slope'])
                    self.adx_minus_slopes.extend([f'adx_{period}_minus_di_slope'])
                    _validate()
        else:
            for period in parameters:
                self.adx_names.extend([f'adx_{period}'])
                self.adx_plus.extend([f'adx_{period}_plus'])
                self.adx_minus.extend([f'adx_{period}_minus'])
                self.adx_slopes.extend([f'adx_{period}_slope'])
                self.adx_plus_slopes.extend([f'adx_{period}_plus_di_slope'])
                self.adx_minus_slopes.extend([f'adx_{period}_minus_di_slope'])
                _validate()
        
    def adx_trend_strength_signals(
        self, 
        strong_threshold: int = 25, 
        weak_threshold: int = 20,
    ):
        
        """
        ADX Trend Strength Signals
        2 = Strong Trend (ADX > 25)
        1 = Weak Trend (ADX < 20) 
        0 = Developing Trend (20 <= ADX <= 25)
        
        Parameters:
        strong_threshold (int): Threshold for strong trend (default: 25)
        weak_threshold (int): Threshold for weak trend (default: 20)

        
        """
        
        for name in self.adx_names:
            self._validate_columns(columns = [name])

            strong_trend = self.data[name] > strong_threshold
            weak_trend = self.data[name] < weak_threshold

            self.signals[f'{name}_trend_strength'] = np.select(
                [strong_trend, weak_trend],
                [2, 1],
                default=0
            )
                    
        return self.signals
    
    def adx_direction_signals(
        self,
    ):
        
        """
        ADX Direction Signals (using +DI and -DI)
        2 = Bullish (+DI > -DI)
        1 = Bearish (-DI > +DI)
        0 = Neutral
            
        """
        
        for name1, name2 in product(self.adx_plus, self.adx_minus):
            self._validate_columns(columns = [name1, name2])
            
            bullish = self.data[name1] > self.data[name2]
            bearish = self.data[name2] > self.data[name1]

            self.signals[f'{name1}_{name2}_direction'] = np.select(
                [bullish, bearish],
                [2, 1],
                default=0
            )
                    
        return self.signals
    
    def adx_di_crossover_signals(
        self,
    ):
        
        """
        +DI/-DI Crossover Signals
        2 = +DI crosses above -DI (Bullish)
        1 = -DI crosses above +DI (Bearish)
        0 = No crossover
        
        """
        
        for name1, name2 in product(self.adx_plus, self.adx_minus):
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
                default=0
            )
                    
        return self.signals
    
    def adx_slope_signals(
        self,
        column: str = 'trend_adx_slope',
    ):
        
        """
        ADX Slope Signals (Momentum of trend strength)
        2 = ADX increasing (trend strengthening)
        1 = ADX decreasing (trend weakening)
        0 = ADX flat
        
        Parameters:
        column (str): Column name for ADX slope
        
        """
        
        for name in self.adx_slopes:
            self._validate_columns(columns = [name])
            
            increasing = self.data[name] > 0
            decreasing = self.data[name] < 0
            
            self.signals[f'{name}_slope'] = np.select(
                [increasing, decreasing],
                [2, 1],
                default=0
            )
                    
        return self.signals
        
    def adx_comprehensive_signals(
        self,
        strong_threshold: int =25,
    ):
        
        """
        Comprehensive ADX Signals (Trend Strength + Direction)
        2 = Strong Bullish Trend (ADX > 25 & +DI > -DI)
        1 = Strong Bearish Trend (ADX > 25 & -DI > +DI)
        0 = Weak/No clear trend
        
        Parameters:
        columns (list[str]): List of column names for ADX, +DI, and -DI
        
        """
        
        for name1, name2, name3 in product(self.adx_names, self.adx_plus, self.adx_minus):
            self._validate_columns(columns = [name1, name2, name3])

            strong_bullish = (
                (self.data[name1] > strong_threshold) & 
                (self.data[name2] > self.data[name3])
            )
            
            strong_bearish = (
                (self.data[name1] > strong_threshold) & 
                (self.data[name3] > self.data[name2])
            )
            
            self.signals[f'{name1}_{name2}_{name3}_comprehensive'] = np.select(
                [strong_bullish, strong_bearish],
                [2, 1],
                default=0
            )
                    
        return self.signals
    
    def generate_all_adx_signals(self): 
        
        """
        Generate all ADX signals
        
        """
        
        self.adx_trend_strength_signals()
        self.adx_direction_signals()
        self.adx_di_crossover_signals()
        self.adx_slope_signals()
        self.adx_comprehensive_signals()

        count_removed_rows = self.signals.shape[0] - self.data.shape[0]
        
        if self.prints:
            print('='*50)
            print('Data Info')
            print(self.signals.info())
            print('='*50)   
            print(f'Shape of data {self.signals.shape}')
            print('='*50)   
            print(f'{count_removed_rows} rows removed')
            print('='*50)

        return self.signals