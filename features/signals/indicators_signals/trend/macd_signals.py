import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

from itertools import product

import warnings
warnings.filterwarnings('ignore')

class ForexMACDSignals:    
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
        parameters: List = None,
        prints = True
        
    ):
        """
        Class for MACD signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        parameters (List): List of parameters for MACD calculation
        
        """
        
        self.prints = prints
        
        if self.prints:
            print("="*50)
            print("MACD SIGNAL GENERATION")
            print("="*50)
            print(" Available Fuctions: \n1 macd_crossover_signals \n2 macd_histogram_signals \n3 macd_zero_line_signals \n4 generate_all_macd_signals")
            print("="*50)
        
        self.close_col = close_col
        self.data = data.copy()
        
        self.signals = pd.DataFrame(
            {self.close_col: self.data[self.close_col]},
            index=self.data.index
        )
        
        self.macds = []
        self.macd_signals = []
        self.macd_hists = []
        self.macd_slopes = []
        self.macd_signal_slopes = []
        
        self.macd_parameters = [12, 26, 9] if parameters is None else parameters
        
        self._validate_columns()
        self._extract_column_names(parameters = self.macd_parameters)
    
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
        
        if is_nested:
            for sublist in parameters:
                fast = sublist[0]
                slow = sublist[1]
                signal = sublist[2]
                self.macds.extend([f'macd_{fast}_{slow}'])
                self.macd_signals.extend([f'macd_signal_{signal}'])
                self.macd_hists.extend([f'macd_hist_fast_{fast}_slow{slow}_sig{signal}'])
                self.macd_slopes.extend([f'macd_{fast}_{slow}_slope'])
                self.macd_signal_slopes.extend([f'macd_signal_{signal}_slope'])
        else:
            fast = parameters[0]
            slow = parameters[1]
            signal = parameters[2]
            self.macds.extend([f'macd_{fast}_{slow}']) 
            self.macd_signals.extend([f'macd_signal_{signal}'])
            self.macd_hists.extend([f'macd_hist_fast_{fast}_slow{slow}_sig{signal}'])
            self.macd_slopes.extend([f'macd_{fast}_{slow}_slope'])
            self.macd_signal_slopes.extend([f'macd_signal_{signal}_slope'])
          
    def macd_crossover_signals(
        self,
    ):
        
        """
        MACD Line vs Signal Line Crossover
        2 = MACD crosses above Signal
        1 = MACD crosses below Signal
        0 = No crossover
        
        Parameters:
        columns (list[str]): List of column names for MACD and Signal
        
        """
        
        for name1, name2 in product(self.macds, self.macd_signals):
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
    
    def macd_histogram_signals(
        self,
    ):
        
        """
        MACD Histogram Signals
        2 = Histogram is positive and increasing (bullish momentum)
        1 = Histogram is negative and decreasing (bearish momentum)
        0 = No clear momentum
        
        Parameters:
        column (str): Column name for MACD histogram
        
        """
        
        for name in self.macd_hists:
            self._validate_columns(columns = [name])

            hist_positive = self.data[name] > 0
            hist_negative = self.data[name] < 0

            self.signals[f'{name}_direction'] = np.select(
                [hist_positive, hist_negative],
                [2, 1],
                default=0
            )
            
            hist_increasing = self.data[name] > self.data[name].shift(1)
            hist_decreasing = self.data[name] < self.data[name].shift(1)

            self.signals[f'{name}_momentum'] = np.select(
                [hist_increasing, hist_decreasing],
                [2, 1],
                default=0
            )
        
        return self.signals
    
    def macd_zero_line_signals(
        self,
    ):
        
        """
        MACD Zero Line Crossover Signals
        2 = MACD crosses above Zero Line (bullish)
        1 = MACD crosses below Zero Line (bearish)
        0 = No crossover
        
        """
        
        for name in self.macds:
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
                default=0
            )
                    
        return self.signals
        
    def generate_all_macd_signals(self):
       
        """
        Generate all MACD signals
       
        """
        
        self.macd_crossover_signals()
        self.macd_histogram_signals()
        self.macd_zero_line_signals()
        
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
       