import pandas as pd
import numpy as np
import talib
from typing import Dict, List, Optional, Union

import warnings
warnings.filterwarnings('ignore')

class ForexMACDSignals:    
    def __init__(
        self, 
        data: pd.DataFrame,
        close_col: str = 'close',
    ):
        """
        Class for MACD signals
        
        Parameters:
        data (pd.DataFrame): DataFrame containing the data    
        close_col (str): Column name for close price
        
        """
        
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
        
        self._validate_columns()
    
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
    
    def macd_crossover_signals(
        self,
        columns = ['trend_macd', 'trend_macd_signal'],
    ):
        
        """
        MACD Line vs Signal Line Crossover
        2 = MACD crosses above Signal
        1 = MACD crosses below Signal
        0 = No crossover
        
        Parameters:
        columns (list[str]): List of column names for MACD and Signal
        
        """
        
        self._validate_columns(columns = columns)
        
        bullish_cross = ( # MACD crosses above Signal
            (self.data[columns[0]] > self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) <= self.data[columns[1]].shift(1))
        )
        
        bearish_cross = ( # MACD crosses below Signal
            (self.data[columns[0]] < self.data[columns[1]]) & 
            (self.data[columns[0]].shift(1) >= self.data[columns[1]].shift(1))
        )
        
        self.signals['macd_crossover'] = np.select(
            [bullish_cross, bearish_cross],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def macd_histogram_signals(
        self,
        column: str = 'trend_macd_hist',
    ):
        
        """
        MACD Histogram Signals
        2 = Histogram is positive and increasing (bullish momentum)
        1 = Histogram is negative and decreasing (bearish momentum)
        0 = No clear momentum
        
        Parameters:
        column (str): Column name for MACD histogram
        
        """

        self._validate_columns(columns = [column])
        
        # Histogram positive/negative
        hist_positive = self.data[column] > 0
        hist_negative = self.data[column] < 0
        
        self.signals['macd_histogram_direction'] = np.select(
            [hist_positive, hist_negative],
            [2, 1],
            default=0
        )
        
        # Histogram momentum
        hist_increasing = self.data[column] > self.data[column].shift(1)
        hist_decreasing = self.data[column] < self.data[column].shift(1)
        
        self.signals['macd_histogram_momentum'] = np.select(
            [hist_increasing, hist_decreasing],
            [2, 1],
            default=0
        )
        
        return self.signals
    
    def macd_zero_line_signals(
        self,
        column: str = 'trend_macd'
    ):
        
        """
        MACD Zero Line Crossover Signals
        2 = MACD crosses above Zero Line (bullish)
        1 = MACD crosses below Zero Line (bearish)
        0 = No crossover
        
        Parameters:
        column (str): Column name for MACD
        
        """
        
        self._validate_columns(columns = [column])
        
        above_zero = (
            (self.data[column] > 0) & 
            (self.data[column].shift(1) <= 0)
        )
        
        below_zero = (
            (self.data[column] < 0) & 
            (self.data[column].shift(1) >= 0)
        )
        
        self.signals['macd_zero_cross'] = np.select(
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
        print(self.signals.tail(10), "\n", self.signals.shape)
        return self.signals
       